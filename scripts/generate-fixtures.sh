#!/usr/bin/env bash
# generate-fixtures.sh — Populate committed fixtures that require downloading large source datasets.
#
# Currently populates:
#   test-data/fixtures/miracl-ko-mini/  (2000-doc deterministic sample of MIRACL-Ko, seed=42)
#   test-data/fixtures/miracl-zh-mini/  (2000-doc deterministic sample of MIRACL-ZH, seed=42)
#
# Requires: scripts/download-miracl-ko.sh / download-miracl-zh.sh to have been run first
# Usage:    scripts/generate-fixtures.sh [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
source "$SCRIPT_DIR/bench-env.sh"
bench_env_init "$REPO_ROOT" "generate-fixtures"

FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

generate_mini_fixture() {
    local lang="$1"        # e.g. "ko" or "zh"
    local lang_upper="$2"  # e.g. "Ko" or "ZH" (used in display strings)
    local download_script="$3"

    local fixture="$REPO_ROOT/test-data/fixtures/miracl-${lang}-mini"
    local source="$REPO_ROOT/test-data/miracl-${lang}"

    if [[ -f "$fixture/corpus.jsonl" && "$FORCE" -eq 0 ]]; then
        echo "[skip] test-data/fixtures/miracl-${lang}-mini — already populated (use --force to regenerate)"
        return
    fi

    if [[ ! -f "$source/corpus.jsonl" ]]; then
        echo "ERROR: MIRACL-${lang_upper} corpus not found at $source/corpus.jsonl" >&2
        echo "       Run: bash $download_script" >&2
        exit 1
    fi

    echo "==> Generating miracl-${lang}-mini (2000 docs, seed=42)..."
    rm -f "$fixture/corpus.jsonl" "$fixture/queries.jsonl"
    rm -rf "$fixture/qrels"
    mkdir -p "$fixture"

    local stage_dir
    stage_dir=$(mktemp -d "$TMPDIR/miracl-${lang}-mini-XXXXXX")

    python3 scripts/beir-eval.py sample \
        --data "$source" \
        --size 2000 \
        --seed 42 \
        --output "$stage_dir"

    mv "$stage_dir/corpus.jsonl" "$fixture/corpus.jsonl"
    mv "$stage_dir/queries.jsonl" "$fixture/queries.jsonl"
    mv "$stage_dir/qrels" "$fixture/qrels"
    rmdir "$stage_dir"

    echo "==> Done. Fixture at test-data/fixtures/miracl-${lang}-mini/"
    echo "    Commit corpus.jsonl, queries.jsonl, qrels/ alongside expected.json."
    echo "    Then run: scripts/calibrate-fixtures.sh miracl-${lang}-mini"
}

generate_mini_fixture "ko" "Ko" "scripts/download-miracl-ko.sh"
generate_mini_fixture "zh" "ZH" "scripts/download-miracl-zh.sh"
