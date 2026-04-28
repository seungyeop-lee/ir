// Post-retrieval filter: evaluate Filter clauses against documents table + document_metadata.
// Applied at every pipeline exit point so each tier returns correctly filtered results.

use crate::db::CollectionDb;
use crate::error::Result;
use crate::types::{Filter, FilterClause, FilterOp, SearchResult};
use std::collections::HashMap;

/// Retain only candidates that pass all clauses in `filter`.
/// Short-circuits immediately when filter is empty.
///
/// Per collection: one SQL batch for built-in fields, one optional batch for meta.* fields.
/// All values are parameter-bound — no injection surface.
pub fn apply(
    candidates: &mut Vec<SearchResult>,
    filter: &Filter,
    dbs: &[CollectionDb],
) -> Result<()> {
    if filter.is_empty() {
        return Ok(());
    }

    let has_meta = filter.clauses.iter().any(|c| c.field.starts_with("meta."));

    // Group candidate paths by collection name
    let mut by_collection: HashMap<&str, Vec<&str>> = HashMap::new();
    for c in candidates.iter() {
        by_collection
            .entry(c.collection.as_str())
            .or_default()
            .push(c.path.as_str());
    }

    // field_maps: (collection, path) → {field → Vec<value>}
    // Vec<value> supports multi-valued keys (e.g. tags → one row per tag).
    let mut field_maps: HashMap<(String, String), HashMap<String, Vec<String>>> = HashMap::new();

    for db in dbs {
        let paths = match by_collection.get(db.name.as_str()) {
            Some(p) => p,
            None => continue,
        };
        if paths.is_empty() {
            continue;
        }

        let conn = db.conn();
        let placeholders = paths.iter().map(|_| "?").collect::<Vec<_>>().join(",");

        // Batch-fetch built-in fields (path, modified_at, created_at)
        {
            let sql = format!(
                "SELECT path, modified_at, created_at \
                 FROM documents \
                 WHERE path IN ({placeholders}) AND active = 1"
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows =
                stmt.query_map(rusqlite::params_from_iter(paths.iter().copied()), |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?;
            for row in rows {
                let (path, modified_at, created_at) = row?;
                let key = (db.name.clone(), path.clone());
                let fields = field_maps.entry(key).or_default();
                fields.insert("modified_at".to_string(), vec![modified_at]);
                fields.insert("created_at".to_string(), vec![created_at]);
                fields.insert("path".to_string(), vec![path]);
            }
        }

        // Batch-fetch frontmatter metadata (only when filter has meta.* clauses)
        if has_meta {
            let sql = format!(
                "SELECT d.path, dm.key, dm.value \
                 FROM document_metadata dm \
                 JOIN documents d ON dm.document_id = d.id \
                 WHERE d.path IN ({placeholders}) AND d.active = 1"
            );
            let mut stmt = conn.prepare(&sql)?;
            let rows =
                stmt.query_map(rusqlite::params_from_iter(paths.iter().copied()), |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })?;
            for row in rows {
                let (path, key, value) = row?;
                let map_key = (db.name.clone(), path);
                let meta_key = format!("meta.{key}");
                field_maps
                    .entry(map_key)
                    .or_default()
                    .entry(meta_key)
                    .or_default()
                    .push(value);
            }
        }
    }

    candidates.retain(|c| {
        let key = (c.collection.clone(), c.path.clone());
        let Some(fields) = field_maps.get(&key) else {
            return false; // doc not found in DB — exclude
        };
        filter
            .clauses
            .iter()
            .all(|clause| eval_clause(clause, fields))
    });

    Ok(())
}

/// Over-fetch multiplier: 3 with no filter (current baseline), 5 with filter active.
/// Applied to `limit` at every stage where a list is capped; clamped to [50, 500] by caller.
pub fn over_fetch_multiplier(filter: &Filter) -> usize {
    if filter.is_empty() { 3 } else { 5 }
}

fn eval_clause(clause: &FilterClause, fields: &HashMap<String, Vec<String>>) -> bool {
    match fields.get(&clause.field) {
        // Multi-valued fields (e.g. tags): any matching value = clause passes
        Some(values) => values.iter().any(|v| match_op(v, clause.op, &clause.value)),
        // meta.* with no rows → false; built-in fields always present once fetched
        None => false,
    }
}

fn match_op(actual: &str, op: FilterOp, expected: &str) -> bool {
    match op {
        FilterOp::Eq => actual == expected,
        FilterOp::Ne => actual != expected,
        // Lexicographic order — correct for UTC RFC3339 dates (uniform format)
        FilterOp::Gt => actual > expected,
        FilterOp::Gte => actual >= expected,
        FilterOp::Lt => actual < expected,
        FilterOp::Lte => actual <= expected,
        // expected is pre-lowercased at parse time (parse_clause in types.rs)
        FilterOp::Contains => actual.to_ascii_lowercase().contains(expected),
        FilterOp::NotContains => !actual.to_ascii_lowercase().contains(expected),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FilterOp;

    fn fields(pairs: &[(&str, &[&str])]) -> HashMap<String, Vec<String>> {
        pairs
            .iter()
            .map(|(k, vs)| (k.to_string(), vs.iter().map(|v| v.to_string()).collect()))
            .collect()
    }

    fn clause(field: &str, op: FilterOp, value: &str) -> FilterClause {
        FilterClause {
            field: field.to_string(),
            op,
            value: value.to_string(),
        }
    }

    // --- match_op ---

    #[test]
    fn eq_case_sensitive() {
        assert!(match_op("rust", FilterOp::Eq, "rust"));
        assert!(!match_op("Rust", FilterOp::Eq, "rust"));
    }

    #[test]
    fn ne_basic() {
        assert!(match_op("go", FilterOp::Ne, "rust"));
        assert!(!match_op("rust", FilterOp::Ne, "rust"));
    }

    #[test]
    fn lex_comparisons() {
        // Dates as UTC RFC3339 sort lexicographically
        assert!(match_op(
            "2026-01-02T00:00:00Z",
            FilterOp::Gt,
            "2026-01-01T00:00:00Z"
        ));
        assert!(match_op(
            "2026-01-01T00:00:00Z",
            FilterOp::Gte,
            "2026-01-01T00:00:00Z"
        ));
        assert!(match_op(
            "2024-01-01T00:00:00Z",
            FilterOp::Lt,
            "2025-01-01T00:00:00Z"
        ));
        assert!(match_op(
            "2025-01-01T00:00:00Z",
            FilterOp::Lte,
            "2025-01-01T00:00:00Z"
        ));
        assert!(!match_op(
            "2024-01-01T00:00:00Z",
            FilterOp::Gt,
            "2025-01-01T00:00:00Z"
        ));
    }

    #[test]
    fn contains_case_insensitive() {
        // expected is pre-lowercased by parse_clause; actual is lowercased at eval time
        assert!(match_op("Rust Language", FilterOp::Contains, "rust"));
        assert!(match_op("RUST", FilterOp::Contains, "rust"));
        assert!(!match_op("go", FilterOp::Contains, "rust"));
    }

    #[test]
    fn not_contains_case_insensitive() {
        assert!(match_op("go", FilterOp::NotContains, "rust"));
        assert!(!match_op("Rust Language", FilterOp::NotContains, "rust"));
    }

    // --- eval_clause ---

    #[test]
    fn eval_single_value_eq() {
        let f = fields(&[("path", &["notes/foo.md"])]);
        assert!(eval_clause(
            &clause("path", FilterOp::Eq, "notes/foo.md"),
            &f
        ));
        assert!(!eval_clause(
            &clause("path", FilterOp::Eq, "notes/bar.md"),
            &f
        ));
    }

    #[test]
    fn eval_missing_field_returns_false() {
        let f = fields(&[("path", &["notes/foo.md"])]);
        // meta.tags not in map → false
        assert!(!eval_clause(&clause("meta.tags", FilterOp::Eq, "rust"), &f));
    }

    #[test]
    fn eval_multi_valued_any_match() {
        // tags = ["rust", "go"] — Eq passes if ANY element matches
        let f = fields(&[("meta.tags", &["rust", "go"])]);
        assert!(eval_clause(&clause("meta.tags", FilterOp::Eq, "rust"), &f));
        assert!(eval_clause(&clause("meta.tags", FilterOp::Eq, "go"), &f));
        assert!(!eval_clause(
            &clause("meta.tags", FilterOp::Eq, "python"),
            &f
        ));
    }

    #[test]
    fn eval_ne_multi_valued_any_semantics() {
        // Ne on ["rust", "go"]: passes if ANY value != expected.
        // A doc tagged ["rust", "go"] passes meta.tags!=rust because "go" != "rust".
        let f = fields(&[("meta.tags", &["rust", "go"])]);
        assert!(eval_clause(&clause("meta.tags", FilterOp::Ne, "rust"), &f));
        // Single-valued: ["rust"] — no value != "rust" → false
        let f_single = fields(&[("meta.tags", &["rust"])]);
        assert!(!eval_clause(
            &clause("meta.tags", FilterOp::Ne, "rust"),
            &f_single
        ));
    }

    #[test]
    fn eval_contains_on_path() {
        let f = fields(&[("path", &["notes/knowledge/foo.md"])]);
        assert!(eval_clause(
            &clause("path", FilterOp::Contains, "knowledge"),
            &f
        ));
        assert!(!eval_clause(
            &clause("path", FilterOp::Contains, "archive"),
            &f
        ));
    }
}
