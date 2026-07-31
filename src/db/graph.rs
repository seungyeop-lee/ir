// kNN document similarity graph over stored chunk embeddings.
// Built at embed time (research: IR_GRAPH_BUILD=1); read at query time for
// graph-expanded retrieval (IR_GRAPH_T0_EXPAND / IR_GRAPH_T1_CONSENSUS).
//
// Edges are doc-level: for each active content hash, every chunk embedding is
// kNN-queried against vectors_vec and hits are aggregated per neighbor hash
// (weight = max cosine similarity across chunk pairs). Top-k neighbor hashes
// are then expanded to doc_id pairs (content-addressed: one hash may back
// multiple documents). Deterministic given the vector table contents.

use crate::error::Result;
use crate::llm::from_bytes;
use rusqlite::Connection;
use std::collections::HashMap;

/// Cap on chunks per document consulted during graph build.
/// Bounds build cost for long documents; chunks are taken in seq order.
const MAX_CHUNKS_PER_DOC: usize = 8;

/// Neighbor rows retained per chunk row during scoring (over-fetch so self and
/// duplicate-hash hits can be dropped before the doc-level top-k). Mirrors the
/// old per-chunk kNN fetch_k.
fn fetch_m(k: usize) -> usize {
    (k + 2) * 3 + MAX_CHUNKS_PER_DOC
}

/// Neighbor-side block size for the blocked dot-product pass. 256 rows of a
/// 768-d f32 matrix ≈ 768KB — sized to stay L2-resident while every source row
/// streams through it.
const SCORE_BLOCK_ROWS: usize = 256;

/// A graph neighbor of a source document, hydrated with document metadata.
pub struct NeighborDoc {
    pub path: String,
    pub title: String,
    pub hash: String,
    pub weight: f64,
}

/// Rebuild doc_graph from scratch. Returns (source_docs, edges) counts.
/// k = neighbors kept per document.
pub fn build(conn: &Connection, k: usize) -> Result<(usize, usize)> {
    // Guard: collection DBs created before graph support (schema init adds this
    // via schema_base.sql; open_rw paths don't run init).
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS doc_graph (
            doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            neighbor_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            weight      REAL NOT NULL,
            PRIMARY KEY (doc_id, neighbor_id)
        );",
    )?;

    // hash → doc_ids for all active docs (multiple paths may share content).
    let hash_docs: HashMap<String, Vec<i64>> = {
        let mut stmt =
            conn.prepare("SELECT hash, id FROM documents WHERE active = 1 ORDER BY id")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        let mut map: HashMap<String, Vec<i64>> = HashMap::new();
        for row in rows {
            let (hash, id) = row?;
            map.entry(hash).or_default().push(id);
        }
        map
    };

    // Embedded chunk seqs per hash (seq order → deterministic chunk selection).
    let hash_seqs: HashMap<String, Vec<i64>> = {
        let mut stmt = conn.prepare("SELECT hash, seq FROM content_vectors ORDER BY hash, seq")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        let mut map: HashMap<String, Vec<i64>> = HashMap::new();
        for row in rows {
            let (hash, seq) = row?;
            let seqs = map.entry(hash).or_default();
            if seqs.len() < MAX_CHUNKS_PER_DOC {
                seqs.push(seq);
            }
        }
        map
    };

    // Deterministic iteration order over source hashes.
    let mut source_hashes: Vec<&String> = hash_docs.keys().collect();
    source_hashes.sort();

    // ── Load phase: all selected chunk embeddings into one in-memory matrix. ──
    // Replaces the old per-chunk sqlite-vec kNN scan (O(chunks × table-rows)
    // with per-row SQL overhead; 91min @ 50k docs) with a blocked exact
    // dot-product pass over unit-normalized rows (~1-3min @ 50k).
    let mut data: Vec<f32> = Vec::new(); // flat row-major, rows × dim
    let mut row_hash: Vec<u32> = Vec::new(); // row → index into hashes below
    let mut hashes: Vec<&String> = Vec::new(); // hash_idx → hash (sorted order)
    let mut dim = 0usize;
    {
        let mut stmt = conn.prepare("SELECT embedding FROM vectors_vec WHERE hash_seq = ?1")?;
        for hash in source_hashes.iter().copied() {
            let Some(seqs) = hash_seqs.get(hash) else {
                continue; // not embedded — no rows, no edges
            };
            let hash_idx = hashes.len() as u32;
            let mut pushed = false;
            for seq in seqs {
                let hash_seq = format!("{hash}_{seq}");
                let emb: Option<Vec<u8>> = stmt.query_row([&hash_seq], |row| row.get(0)).ok();
                let Some(blob) = emb else { continue };
                let mut v = from_bytes(&blob);
                if dim == 0 {
                    dim = v.len();
                }
                if v.len() != dim {
                    continue; // mixed-dim vectors (model change mid-collection)
                }
                // Unit-normalize so dot == cosine similarity.
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm <= 0.0 {
                    continue;
                }
                v.iter_mut().for_each(|x| *x /= norm);
                data.extend_from_slice(&v);
                row_hash.push(hash_idx);
                pushed = true;
            }
            if pushed {
                hashes.push(hash);
            }
        }
    }
    let n_rows = row_hash.len();

    // ── Scoring phase: per source hash, max cosine per neighbor hash, top-k. ──
    // Parallel over contiguous source-hash shards; deterministic because each
    // hash's result is independent and shards are concatenated in order.
    let m = fetch_m(k);
    let n_hashes = hashes.len();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(n_hashes.max(1));
    // hash_idx → contiguous row range (rows were pushed in hash order).
    let mut hash_rows: Vec<(usize, usize)> = vec![(0, 0); n_hashes];
    {
        let mut start = 0usize;
        for r in 0..n_rows {
            if r + 1 == n_rows || row_hash[r + 1] != row_hash[r] {
                hash_rows[row_hash[r] as usize] = (start, r + 1);
                start = r + 1;
            }
        }
    }

    // Per hash: ranked top-k (neighbor_hash_idx, weight), hash-tiebroken.
    let mut ranked_per_hash: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n_hashes];
    let shard_size = n_hashes.div_ceil(threads.max(1));
    std::thread::scope(|scope| {
        let mut remaining: &mut [Vec<(u32, f64)>] = &mut ranked_per_hash;
        let mut shard_start = 0usize;
        let data = &data;
        let row_hash = &row_hash;
        let hash_rows = &hash_rows;
        let hashes = &hashes;
        while !remaining.is_empty() {
            let len = remaining.len().min(shard_size);
            let (shard, rest) = remaining.split_at_mut(len);
            remaining = rest;
            let start = shard_start;
            shard_start += len;
            scope.spawn(move || {
                for (off, out) in shard.iter_mut().enumerate() {
                    let hash_idx = (start + off) as u32;
                    let (r0, r1) = hash_rows[hash_idx as usize];
                    // Per chunk row: top-m neighbor rows by dot product,
                    // then fold into per-neighbor-hash max weight.
                    let mut neighbor_weights: HashMap<u32, f64> = HashMap::new();
                    for r in r0..r1 {
                        let src = &data[r * dim..(r + 1) * dim];
                        // (weight, neighbor_row) min-kept heap emulated with a
                        // sorted vec of size m — m is small (≈40).
                        let mut top: Vec<(f32, u32)> = Vec::with_capacity(m + 1);
                        let mut block = 0usize;
                        while block < n_rows {
                            let end = (block + SCORE_BLOCK_ROWS).min(n_rows);
                            for c in block..end {
                                if row_hash[c] == hash_idx {
                                    continue; // self / same content hash
                                }
                                let dst = &data[c * dim..(c + 1) * dim];
                                let dot = dot_f32(src, dst);
                                if dot <= 0.0 {
                                    continue;
                                }
                                if top.len() < m {
                                    top.push((dot, c as u32));
                                    if top.len() == m {
                                        top.sort_by(|a, b| {
                                            a.0.partial_cmp(&b.0)
                                                .unwrap_or(std::cmp::Ordering::Equal)
                                        });
                                    }
                                } else if dot > top[0].0 {
                                    // replace the current minimum, keep sorted
                                    let pos = top
                                        .partition_point(|&(w, _)| w < dot)
                                        .saturating_sub(1);
                                    top.remove(0);
                                    top.insert(pos, (dot, c as u32));
                                }
                            }
                            block = end;
                        }
                        for &(w, c) in &top {
                            let nb = row_hash[c as usize];
                            let entry = neighbor_weights.entry(nb).or_insert(0.0);
                            if f64::from(w) > *entry {
                                *entry = f64::from(w);
                            }
                        }
                    }
                    if neighbor_weights.is_empty() {
                        continue;
                    }
                    let mut ranked: Vec<(u32, f64)> = neighbor_weights.into_iter().collect();
                    ranked.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| hashes[a.0 as usize].cmp(hashes[b.0 as usize]))
                    });
                    ranked.truncate(k);
                    *out = ranked;
                }
            });
        }
    });

    // ── Emit phase: expand hash-level edges to doc_id pairs (original order). ──
    let mut edges: Vec<(i64, i64, f64)> = Vec::new();
    let mut source_count = 0usize;
    for (hash_idx, ranked) in ranked_per_hash.iter().enumerate() {
        if ranked.is_empty() {
            continue;
        }
        source_count += 1;
        let source_ids = &hash_docs[hashes[hash_idx]];
        for &(neighbor_idx, weight) in ranked {
            let Some(neighbor_ids) = hash_docs.get(hashes[neighbor_idx as usize]) else {
                continue;
            };
            for &doc_id in source_ids {
                for &neighbor_id in neighbor_ids {
                    edges.push((doc_id, neighbor_id, weight));
                }
            }
        }
    }

    // Rewrite atomically.
    let tx = conn.unchecked_transaction()?;
    tx.execute("DELETE FROM doc_graph", [])?;
    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO doc_graph (doc_id, neighbor_id, weight) VALUES (?1, ?2, ?3)",
        )?;
        for (doc_id, neighbor_id, weight) in &edges {
            stmt.execute(rusqlite::params![doc_id, neighbor_id, weight])?;
        }
    }
    tx.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('graph_k', ?1)",
        [k.to_string()],
    )?;
    tx.commit()?;

    Ok((source_count, edges.len()))
}

/// Batch-fetch graph neighbors for the given source paths.
/// Returns source path → neighbors. Empty map when the graph is absent
/// (collections indexed before graph support, or never built).
pub fn neighbors_for_paths(conn: &Connection, paths: &[&str]) -> HashMap<String, Vec<NeighborDoc>> {
    if paths.is_empty() {
        return HashMap::new();
    }
    let placeholders = paths.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT d.path, n.path, n.title, n.hash, g.weight
         FROM doc_graph g
         JOIN documents d ON d.id = g.doc_id
         JOIN documents n ON n.id = g.neighbor_id
         WHERE d.path IN ({placeholders}) AND d.active = 1 AND n.active = 1"
    );
    // Missing doc_graph table (pre-graph DB opened read-write) → empty result.
    let mut stmt = match conn.prepare(&sql) {
        Ok(s) => s,
        Err(_) => return HashMap::new(),
    };
    let rows = match stmt.query_map(rusqlite::params_from_iter(paths.iter().copied()), |row| {
        Ok((
            row.get::<_, String>(0)?,
            NeighborDoc {
                path: row.get(1)?,
                title: row.get(2)?,
                hash: row.get(3)?,
                weight: row.get(4)?,
            },
        ))
    }) {
        Ok(r) => r,
        Err(_) => return HashMap::new(),
    };
    let mut map: HashMap<String, Vec<NeighborDoc>> = HashMap::new();
    for row in rows.flatten() {
        map.entry(row.0).or_default().push(row.1);
    }
    map
}

/// Dot product with 8 independent accumulators — breaks the serial FP
/// dependency chain so LLVM can vectorize (NEON/AVX). The build's hot loop.
#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let chunks = a.len() / 8;
    for i in 0..chunks {
        let ai = &a[i * 8..i * 8 + 8];
        let bi = &b[i * 8..i * 8 + 8];
        for j in 0..8 {
            acc[j] += ai[j] * bi[j];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// True when this collection has a built graph (any edges present).
pub fn has_graph(conn: &Connection) -> bool {
    conn.query_row("SELECT 1 FROM doc_graph LIMIT 1", [], |_| Ok(()))
        .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::to_bytes;
    use rusqlite::Connection;

    fn open_test_db() -> Connection {
        crate::db::ensure_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE vectors_vec USING vec0(
                hash_seq TEXT PRIMARY KEY,
                embedding float[4] distance_metric=cosine
             );",
        )
        .unwrap();
        conn.execute_batch(include_str!("schema_base.sql")).unwrap();
        conn
    }

    fn add_doc(conn: &Connection, path: &str, hash: &str, emb: &[f32]) {
        conn.execute(
            "INSERT INTO content (hash, doc, created_at) VALUES (?1, 'text', '2024-01-01')",
            [hash],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO documents (path, title, hash, created_at, modified_at, active)
             VALUES (?1, ?1, ?2, '2024-01-01', '2024-01-01', 1)",
            rusqlite::params![path, hash],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
             VALUES (?1, 0, 0, 'test', '2024-01-01')",
            [hash],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?1, ?2)",
            rusqlite::params![format!("{hash}_0"), to_bytes(emb)],
        )
        .unwrap();
    }

    #[test]
    fn build_creates_topk_edges_excluding_self() {
        let conn = open_test_db();
        // a and b are near-identical; c is orthogonal.
        add_doc(&conn, "a.md", "hasha", &[1.0, 0.0, 0.0, 0.0]);
        add_doc(&conn, "b.md", "hashb", &[0.99, 0.14, 0.0, 0.0]);
        add_doc(&conn, "c.md", "hashc", &[0.0, 0.0, 1.0, 0.0]);

        let (docs, edges) = build(&conn, 2).unwrap();
        // c is orthogonal to a and b (cosine 0) → no positive-weight edges for it.
        assert_eq!(docs, 2);
        assert!(edges >= 2, "a and b should link to each other, got {edges}");

        let nbrs = neighbors_for_paths(&conn, &["a.md"]);
        let a_nbrs = nbrs.get("a.md").unwrap();
        assert!(a_nbrs.iter().all(|n| n.path != "a.md"), "no self-edges");
        // b must be a's strongest neighbor.
        let best = a_nbrs
            .iter()
            .max_by(|x, y| x.weight.partial_cmp(&y.weight).unwrap())
            .unwrap();
        assert_eq!(best.path, "b.md");
        assert!(best.weight > 0.9);
    }

    #[test]
    fn build_is_idempotent() {
        let conn = open_test_db();
        add_doc(&conn, "a.md", "hasha", &[1.0, 0.0, 0.0, 0.0]);
        add_doc(&conn, "b.md", "hashb", &[0.9, 0.43, 0.0, 0.0]);

        let (_, e1) = build(&conn, 5).unwrap();
        let (_, e2) = build(&conn, 5).unwrap();
        assert_eq!(e1, e2, "rebuild should produce identical edge count");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM doc_graph", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count as usize, e2);
    }

    #[test]
    fn neighbors_missing_table_returns_empty() {
        crate::db::ensure_sqlite_vec();
        let conn = Connection::open_in_memory().unwrap();
        // No schema at all — must not error.
        assert!(neighbors_for_paths(&conn, &["a.md"]).is_empty());
        assert!(!has_graph(&conn));
    }

    #[test]
    fn build_matches_bruteforce_reference() {
        let conn = open_test_db();
        // Deterministic pseudo-random 4-d vectors (LCG), 40 docs.
        let mut state = 0x243f_6a88u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (u32::MAX >> 1) as f32) - 1.0
        };
        let mut vecs: Vec<[f32; 4]> = Vec::new();
        for i in 0..40 {
            let v = [next(), next(), next(), next()];
            add_doc(&conn, &format!("d{i:02}.md"), &format!("h{i:02}"), &v);
            vecs.push(v);
        }
        let k = 5;
        build(&conn, k).unwrap();

        // Reference: unit-normalize, all-pairs cosine, top-k per doc with
        // hash tie-break, positive weights only.
        let norm = |v: &[f32; 4]| {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            [v[0] / n, v[1] / n, v[2] / n, v[3] / n]
        };
        let unit: Vec<[f32; 4]> = vecs.iter().map(norm).collect();
        for i in 0..40 {
            let mut sims: Vec<(String, f32)> = (0..40)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f32 = (0..4).map(|x| unit[i][x] * unit[j][x]).sum();
                    (format!("h{j:02}"), d)
                })
                .filter(|(_, d)| *d > 0.0)
                .collect();
            sims.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap()
                    .then_with(|| a.0.cmp(&b.0))
            });
            sims.truncate(k);
            let expected: Vec<&String> = sims.iter().map(|(h, _)| h).collect();

            let got = neighbors_for_paths(&conn, &[&format!("d{i:02}.md")]);
            let mut got_nbrs: Vec<(String, f64)> = got
                .get(&format!("d{i:02}.md"))
                .map(|ns| ns.iter().map(|n| (n.hash.clone(), n.weight)).collect())
                .unwrap_or_default();
            got_nbrs.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap()
                    .then_with(|| a.0.cmp(&b.0))
            });
            let got_hashes: Vec<&String> = got_nbrs.iter().map(|(h, _)| h).collect();
            assert_eq!(got_hashes, expected, "doc d{i:02} neighbor set/order");
        }
    }

    #[test]
    fn unembedded_docs_produce_no_edges() {
        let conn = open_test_db();
        conn.execute(
            "INSERT INTO content (hash, doc, created_at) VALUES ('h1', 'text', '2024-01-01')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO documents (path, title, hash, created_at, modified_at, active)
             VALUES ('lone.md', 'lone', 'h1', '2024-01-01', '2024-01-01', 1)",
            [],
        )
        .unwrap();
        let (docs, edges) = build(&conn, 5).unwrap();
        assert_eq!((docs, edges), (0, 0));
        assert!(!has_graph(&conn));
    }
}
