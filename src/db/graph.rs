// kNN document similarity graph over stored chunk embeddings.
// Built at embed time (research: IR_GRAPH_BUILD=1); read at query time for
// graph-expanded retrieval (IR_GRAPH_T0_EXPAND / IR_GRAPH_T1_CONSENSUS).
//
// Edges are doc-level: for each active content hash, every chunk embedding is
// kNN-queried against vectors_vec and hits are aggregated per neighbor hash
// (weight = max cosine similarity across chunk pairs). Top-k neighbor hashes
// are then expanded to doc_id pairs (content-addressed: one hash may back
// multiple documents). Deterministic given the vector table contents.

use crate::db::vectors;
use crate::error::Result;
use crate::llm::from_bytes;
use rusqlite::Connection;
use std::collections::HashMap;

/// Cap on chunks per document consulted during graph build.
/// Bounds build cost for long documents; chunks are taken in seq order.
const MAX_CHUNKS_PER_DOC: usize = 8;

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

    // Over-fetch so self-chunks and duplicate-hash hits can be filtered out.
    let fetch_k = ((k + 2) * 3 + MAX_CHUNKS_PER_DOC).min(vectors::KNN_MAX);

    let mut edges: Vec<(i64, i64, f64)> = Vec::new();
    let mut source_count = 0usize;

    for (i, hash) in source_hashes.iter().copied().enumerate() {
        let Some(seqs) = hash_seqs.get(hash) else {
            continue; // not embedded — no edges for this doc
        };

        // Aggregate neighbor hash → max cosine similarity across this doc's chunks.
        let mut neighbor_weights: HashMap<String, f64> = HashMap::new();
        for seq in seqs {
            let hash_seq = format!("{hash}_{seq}");
            let emb: Option<Vec<u8>> = conn
                .query_row(
                    "SELECT embedding FROM vectors_vec WHERE hash_seq = ?1",
                    [&hash_seq],
                    |row| row.get(0),
                )
                .ok();
            let Some(blob) = emb else { continue };
            let query = from_bytes(&blob);

            for hit in vectors::knn(conn, &query, fetch_k)? {
                let neighbor_hash = match hit.hash_seq.rsplit_once('_') {
                    Some((h, _)) => h,
                    None => hit.hash_seq.as_str(),
                };
                // Skip self and exact-duplicate content (degenerate weight≈1.0 edges).
                if neighbor_hash == hash.as_str() {
                    continue;
                }
                let weight = 1.0 - hit.distance;
                if weight <= 0.0 {
                    continue;
                }
                let entry = neighbor_weights
                    .entry(neighbor_hash.to_string())
                    .or_insert(0.0);
                if weight > *entry {
                    *entry = weight;
                }
            }
        }

        if neighbor_weights.is_empty() {
            continue;
        }
        source_count += 1;

        // Top-k neighbor hashes by weight (ties broken by hash for determinism).
        let mut ranked: Vec<(String, f64)> = neighbor_weights.into_iter().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.truncate(k);

        // Expand hash-level edges to doc_id pairs.
        let source_ids = &hash_docs[hash];
        for (neighbor_hash, weight) in &ranked {
            let Some(neighbor_ids) = hash_docs.get(neighbor_hash) else {
                continue; // vectors for inactive/removed docs
            };
            for &doc_id in source_ids {
                for &neighbor_id in neighbor_ids {
                    edges.push((doc_id, neighbor_id, *weight));
                }
            }
        }

        if (i + 1) % 2000 == 0 {
            eprintln!("  graph: {}/{} docs", i + 1, source_hashes.len());
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
