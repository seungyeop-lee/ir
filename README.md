# ir

[![crates.io](https://img.shields.io/crates/v/ir-search.svg)](https://crates.io/crates/ir-search)
[![CI](https://github.com/vlwkaos/ir/actions/workflows/ci.yml/badge.svg)](https://github.com/vlwkaos/ir/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[ENG](README.md) | [한국어](README.ko.md) | [中文](README.zh.md)

Local semantic search for markdown knowledge bases. BM25 + vector + LLM reranking, entirely on your machine — one SQLite file per collection, models kept warm by a persistent daemon, all LLM outputs cached.

```bash
brew install vlwkaos/tap/ir          # macOS
cargo install ir-search              # any platform (binary name: ir)
```

```bash
ir collection add notes ~/notes      # register a collection
ir sync notes                        # index text + embed vectors
ir search "memory safety in rust"    # search (daemon auto-starts)
```

BM25 search works with no models at all. Vector/hybrid search downloads models automatically from HuggingFace on first use. Requires Rust 1.80+ if building from source; Metal is linked automatically on macOS, and Linux GPU backends are opt-in (`--features llama-cuda|llama-rocm|llama-vulkan`).

## How it searches

```
query → BM25 (instant) → strong signal? → done
      → hybrid fusion 0.80·vec + 0.20·bm25 → strong signal? → done
      → query expansion (lex/vec/hyde) → RRF → LLM rerank
```

Each tier runs only when the previous one isn't confident, so easy queries stay at millisecond latency and hard queries get the full LLM treatment. Expander outputs and reranker scores are cached in SQLite — repeated queries skip inference entirely.

## Measured quality (v0.17, nDCG@10)

| Corpus | BM25 only | Hybrid fusion | Full pipeline |
|---|---|---|---|
| NFCorpus (en, 3.6k docs, 323 q) | 0.31 | 0.39 | 0.39 |
| FiQA (en, 57.6k docs, 648 q) | 0.24 | 0.40 | — |
| MIRACL-ko 50k sample (ko preprocessor, 213 q) | 0.73 | 0.92 | **0.96** |
| Allganize RAG-eval-KO (ko, 1.4k pages, 298 q) | 0.70 | 0.69 | 0.72 |

Hybrid fusion needs only the 300M embedder (~50–280ms/query warm). The full pipeline adds the expander + reranker on queries that escalate. Korean numbers require the `ko` preprocessor (see below) — without it Korean BM25 is near zero.

## Versions at a glance

- **≤ 0.15** — core pipeline, daemon, MCP, CJK preprocessors.
- **0.16** — `ir sync` (one command for index + embed), self-healing incremental updates: deleted files are hard-removed and moved/restored content reuses cached vectors.
- **0.17** — research infrastructure for graph-expanded retrieval and an optional HNSW ANN index, plus a much faster benchmark toolchain. **All of it is disabled by default and changes nothing about search behavior** — these are opt-in experiments, not baked-in features. Collection DBs gain two empty tables on first write; databases remain fully compatible in both directions with 0.16.

## Documentation

<details>
<summary><strong>Models</strong></summary>

Models download automatically from HuggingFace Hub on first use (cache: `~/.cache/huggingface/`). `HF_HUB_OFFLINE=1` disables downloads.

| Model | Required for |
|---|---|
| [EmbeddingGemma 300M](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) | vector / hybrid search |
| [Qwen3-Reranker 0.6B](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF) | reranking (optional) |
| [qmd-query-expansion 1.7B](https://huggingface.co/tobil/qmd-query-expansion-1.7B) | query expansion (optional) |
| [BGE-M3](https://huggingface.co/ggml-org/bge-m3-Q8_0-GGUF) | Korean-optimized embedding alternative |

**Local models / overrides:**

```bash
export IR_MODEL_DIRS="$HOME/my-models"
export IR_EMBEDDING_MODEL="$HOME/my-models/embeddinggemma-300M-Q8_0.gguf"
export IR_RERANKER_MODEL="$HOME/my-models/qwen3-reranker-0.6b-q8_0.gguf"
export IR_EXPANDER_MODEL="$HOME/my-models/qmd-query-expansion-1.7B-q4_k_m.gguf"
```

`IR_*_MODEL` accepts a `.gguf` path, a directory containing a known model, or a HuggingFace repo ID. Search order: env → `IR_MODEL_DIRS` → `~/local-models/` → `~/.cache/ir/models/` → HF Hub. `IR_COMBINED_MODEL` (single model for expand+rerank) is opt-in for experiments only. Switching embedding models requires `ir embed --force`.

**Config directory:**

```bash
export IR_CONFIG_DIR="~/vault/.config/ir"   # portable; supports ~ and $VAR
```

Precedence: `IR_CONFIG_DIR` → `XDG_CONFIG_HOME/ir` (deprecated) → `~/.config/ir`.

**GPU:** `IR_GPU_LAYERS=0` forces CPU; `IR_GPU_LAYERS=N` partial offload.

</details>

<details>
<summary><strong>Usage</strong></summary>

**Collections & indexing:**

```bash
ir collection add notes ~/notes
ir collection ls
ir collection rm notes
ir status                    # index health per collection

ir sync [notes] [--force]    # text index + embeddings (the default maintenance command)
ir update [notes] [--force]  # text index only — fast, no models
ir embed [notes] [--force]   # vector repair / re-embedding
```

Indexing is incremental and content-addressed (SHA-256): only changed files are reprocessed, identical content is deduplicated, deleted files are removed, and moved/restored content reuses cached vectors without re-inference.

**Search:**

```bash
ir search "memory safety in rust"                 # hybrid (default)
ir search "sqlite architecture" --mode bm25       # no models
ir search "async patterns" --mode vector
ir search "error handling" -c notes --min-score 0.4

ir search "ownership" --json | --md | --files | --full | --chunk | --quiet
ir search "design" -f "modified_at>=2026-01-01" -f "meta.tags=rust"
```

Filter clauses (`-f`, repeatable, ANDed): fields `path`, `modified_at`, `created_at`, `meta.<name>`; ops `=` `!=` `>` `>=` `<` `<=` `~` `!~`. Dates normalize to UTC RFC3339. Multi-valued frontmatter fields match if **any** element satisfies the clause (including `!=`).

**Retrieve documents:**

```bash
ir get "2026/Daily/2026-04-07.md"              # exact → suffix → substring match
ir get "2026-04-07" -c periodic --section "Log" --max-chars 3000
ir multi-get "a.md" "b.md" --json               # {found, not_found}
```

**Daemon:**

```bash
ir daemon start|stop|status   # auto-starts on first search
```

Warm queries round-trip the Unix socket in ~30ms. On a cold start the first query can return BM25 results immediately while models load in the background.

</details>

<details>
<summary><strong>Korean / Japanese / Chinese preprocessors</strong></summary>

CJK text needs morphological tokenization before BM25 — without it, agglutinated words never match morpheme-level queries (Korean BM25 goes from ~0.00 to useful). The same preprocessor runs at index and query time.

```bash
ir preprocessor install ko    # lindera + ko-dic (official binaries; macOS/Linux)
ir preprocessor install ja    # lindera + ipadic
ir preprocessor install zh    # lindera + jieba
ir preprocessor bind ko wiki  # wire to a collection and re-index
```

Binding `ko` also writes the measured Korean routing default (`fused_strong_product: 0.05`) to that collection; explicit `routing:` config always wins. Per-collection routing overrides (`fused_strong_floor/product`, `bm25_strong_floor/gap`) live in `config.yml` and apply when all searched collections agree.

Any executable can be a preprocessor: UTF-8 lines on stdin → 0-or-1 tokenized lines on stdout, stays alive between lines, passes ASCII-only single words through unchanged. Lindera throughput: ~5,600 Korean docs/s on M-series.

**Why it matters** (MIRACL-Korean):

| preprocessor | BM25 nDCG@10 |
|---|---|
| none | 0.00 |
| lindera (`ko`) | 0.73 (50k-doc sample) |

</details>

<details>
<summary><strong>MCP server — Claude Desktop / Claude Code</strong></summary>

```json
{ "mcpServers": { "ir": { "command": "ir", "args": ["mcp"] } } }
```

Tools: `search` (with `mode`, `limit`, `min_score`, `collections`, `filter`), `get`, `multi_get`, `status`, `update`.

HTTP mode for remote/multi-client setups:

```bash
ir mcp --http 3620 [--cors '*' | --cors 'https://app.example.com']
```

> HTTP mode is unauthenticated and binds all interfaces — trusted networks only.

</details>

<details>
<summary><strong>Benchmarks & reproduction</strong></summary>

All numbers above are reproducible with the shipped harness:

```bash
scripts/bench.sh nfcorpus            # full per-mode table, cached per git hash
scripts/bench.sh miracl-ko --size 50000 --seed 42
bash scripts/preship.sh              # stability / speed / quality gate on fixtures
```

Runs are resumable (per-query progress survives crashes) and guarded by a memory watchdog on macOS. Historical BEIR results (older pipeline config): reranking added up to +14.5% nDCG@10 over pure vector on ArguAna; fusion alone was not significantly better than pure vector on English corpora — the reranker is where tier-2 value lives.

v0.17 ships experimental, **off-by-default** research infrastructure explored on these corpora: a document-similarity graph used to widen the reranker's candidate pool (significant on sparse-result corpora), and an optional HNSW index (usearch) for approximate kNN that reached 99.2% top-10 overlap with exact search at nDCG@10 identical to exact (MIRACL-ko 50k validation). These change no default behavior; see `CHANGELOG.md` for details and measured results.

</details>

<details>
<summary><strong>vs qmd</strong></summary>

ir is a Rust port of [qmd](https://github.com/tobi/qmd) with a different storage model and a persistent daemon.

| | qmd | ir |
|---|---|---|
| Storage | single SQLite | per-collection SQLite (`rm name.sqlite` deletes) |
| Process model | spawn per query | daemon keeps models warm |
| LLM cache | reranker scores | reranker scores + expander outputs |
| Cold / warm query (M4 Max) | 9.5s / 840ms | **3.0s / 30ms** |

</details>

<details>
<summary><strong>Development & schema</strong></summary>

```bash
cargo build [--release]
cargo test                   # no models required
cargo test -- --ignored      # model-dependent tests
```

Per-collection schema: `content` (hash → text), `documents`, `documents_fts` (FTS5), `vectors_vec` (sqlite-vec, cosine), `content_vectors` (chunk metadata), `llm_cache` (reranker scores), `document_metadata` (frontmatter), `meta` — plus empty-by-default research tables (`doc_graph`, `ann_keys`) as of 0.17. Global `expander_cache.sqlite` caches expansion outputs. See [research/pipeline.md](research/pipeline.md) for the staged-async daemon design.

</details>

## License

[MIT](LICENSE)
