// Preprocessor Plugin Protocol
//
// Any executable that:
// 1. Reads UTF-8 lines from stdin
// 2. Writes 0 or 1 UTF-8 lines to stdout per input line (0 if all tokens filtered)
// 3. Flushes stdout after each line (or after producing no output for a line)
// 4. Passes ASCII-only single-word lines through unchanged (required for sentinel protocol)
// 5. Stays alive between lines (no exit-per-line)
//
// ir uses a sentinel-based protocol to handle case (2): after each content line,
// ir also sends SENTINEL (ASCII-only). The subprocess echoes SENTINEL unchanged.
// ir reads until it sees SENTINEL, so a dropped content line doesn't deadlock.
//
// Register: ir preprocessor add <alias> <command>
// Bind:     ir collection add <name> <path> --preprocessor <alias>
//
// Examples:
//   ir preprocessor install ko   (installs lindera-tokenize, registers as "ko")
//   ir preprocessor add ja "mecab -Owakati"
//   ir collection add wiki ~/wiki --preprocessor ko
//
// Subprocess lifetime: stays alive for batch indexing; spawned per-query for search.
// Lindera startup: <10ms (Rust binary, embedded dictionary).

use crate::config::expand_path;
use crate::error::Result;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

// ^ ASCII-only, passes through all lindera language filters unchanged (POS = SL/foreign word)
const SENTINEL: &str = "IRSENTINEL";

pub struct PreprocessHandle {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
}

impl PreprocessHandle {
    /// Spawn a preprocessor subprocess from a command string (e.g. "mecab -Owakati").
    /// Returns None on spawn failure (logs warning).
    pub fn spawn(cmd_str: &str) -> Option<Self> {
        // ! paths with spaces unsupported; commands must be simple tokens (e.g. "mecab -Owakati")
        let mut parts = cmd_str.split_whitespace();
        let raw_program = parts.next()?;
        let expanded = expand_path(raw_program);
        let program = expanded.as_os_str();
        let args: Vec<&str> = parts.collect();

        match Command::new(program)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
        {
            Ok(mut child) => {
                let stdin = BufWriter::new(child.stdin.take()?);
                let stdout = BufReader::new(child.stdout.take()?);
                Some(Self {
                    child,
                    stdin,
                    stdout,
                })
            }
            Err(e) => {
                eprintln!("warning: failed to spawn preprocessor '{cmd_str}': {e}");
                eprintln!(
                    "hint: run `ir preprocessor install <lang>` to reinstall from official lindera releases"
                );
                None
            }
        }
    }

    pub fn process_line(&mut self, line: &str) -> Result<String> {
        if line.trim().is_empty() {
            return Ok(String::new());
        }
        // Flush content + sentinel together; sentinel unblocks read_line() even when
        // the content line produced no output (see protocol spec at top of file).
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.write_all(SENTINEL.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let mut parts = Vec::new();
        let mut buf = String::new();
        loop {
            buf.clear();
            let n = self.stdout.read_line(&mut buf)?;
            if n == 0 {
                break; // subprocess exited unexpectedly; treat as end of output
            }
            let s = buf.trim_end_matches('\n');
            if s == SENTINEL {
                break;
            }
            parts.push(s.to_string());
        }
        Ok(parts.join(" "))
    }

    /// Process multi-line text: split on '\n', process each line, rejoin.
    pub fn process_text(&mut self, text: &str) -> Result<String> {
        let lines: Vec<&str> = text.split('\n').collect();
        let mut out = Vec::with_capacity(lines.len());
        for line in lines {
            out.push(self.process_line(line)?);
        }
        Ok(out.join("\n"))
    }
}

impl Drop for PreprocessHandle {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Chain of preprocessors — output of one feeds into the next.
/// Each specializes in one language and passes other text through unchanged.
/// FTS5's porter unicode61 tokenizer always runs after as the final stage.
pub struct PreprocessChain {
    handles: Vec<PreprocessHandle>,
}

impl PreprocessChain {
    /// Spawn a chain from a list of command strings.
    /// Handles that fail to spawn are skipped with a warning.
    pub fn spawn(commands: &[String]) -> Self {
        let handles = commands
            .iter()
            .filter_map(|cmd| PreprocessHandle::spawn(cmd))
            .collect();
        Self { handles }
    }

    /// Pipe text through all handles in sequence.
    pub fn process_text(&mut self, text: &str) -> Result<String> {
        let mut current = text.to_string();
        for handle in &mut self.handles {
            current = handle.process_text(&current)?;
        }
        Ok(current)
    }

    /// Returns true if at least one preprocessor handle was successfully spawned.
    pub fn is_active(&self) -> bool {
        !self.handles.is_empty()
    }
}

/// Preprocess a query through a command chain. Falls back to raw query on spawn failure or I/O error.
#[cfg(test)]
pub fn preprocess_query(query: &str, commands: &[String]) -> String {
    if commands.is_empty() {
        return query.to_string();
    }
    let mut chain = PreprocessChain::spawn(commands);
    if !chain.is_active() {
        eprintln!(
            "warning: preprocessor configured ({}) but failed to start; using raw query",
            commands.join(", ")
        );
        return query.to_string();
    }
    chain
        .process_text(query)
        .unwrap_or_else(|_| query.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_preprocessor_with_cat() {
        let mut chain = PreprocessChain::spawn(&["cat".to_string()]);
        assert!(chain.is_active(), "cat should spawn successfully");
        let out = chain.process_text("hello world").unwrap();
        assert_eq!(out, "hello world");
    }

    #[test]
    fn chain_with_invalid_command_falls_back() {
        let cmds = vec!["__nonexistent_command_xyz__".to_string()];
        let chain = PreprocessChain::spawn(&cmds);
        assert!(!chain.is_active(), "invalid command should not spawn");
    }

    #[test]
    fn multiline_text_preserved() {
        let mut chain = PreprocessChain::spawn(&["cat".to_string()]);
        assert!(chain.is_active());
        let input = "line one\nline two\nline three";
        let out = chain.process_text(input).unwrap();
        assert_eq!(out, input);
    }

    #[cfg(unix)]
    #[test]
    fn subprocess_is_invoked() {
        // Verifies the pipe protocol works. rev buffers in pipe mode on macOS — use cat.
        let mut chain = PreprocessChain::spawn(&["cat".to_string()]);
        assert!(chain.is_active());
        let out = chain.process_text("hello world").unwrap();
        assert_eq!(out, "hello world");
    }

    #[cfg(unix)]
    #[test]
    fn chain_pipes_through_multiple() {
        let cmds = vec!["cat".to_string(), "cat".to_string()];
        let mut chain = PreprocessChain::spawn(&cmds);
        assert!(chain.is_active());
        let out = chain.process_text("hello world").unwrap();
        assert_eq!(out, "hello world");
    }

    #[cfg(unix)]
    #[test]
    fn preprocess_query_applies_chain() {
        let cmds = vec!["cat".to_string()];
        let out = preprocess_query("hello world", &cmds);
        assert_eq!(out, "hello world");
    }

    #[cfg(unix)]
    #[test]
    fn preprocess_query_empty_commands_returns_raw() {
        let out = preprocess_query("HELLO WORLD", &[]);
        assert_eq!(out, "HELLO WORLD");
    }

    #[cfg(unix)]
    #[test]
    fn preprocess_query_invalid_command_returns_raw() {
        let cmds = vec!["__nonexistent_command_xyz__".to_string()];
        let out = preprocess_query("HELLO WORLD", &cmds);
        assert_eq!(out, "HELLO WORLD");
    }

    #[cfg(unix)]
    #[test]
    fn chain_skips_invalid_handles() {
        let cmds = vec!["__nonexistent_command_xyz__".to_string(), "cat".to_string()];
        let mut chain = PreprocessChain::spawn(&cmds);
        assert!(chain.is_active());
        let out = chain.process_text("hello world").unwrap();
        assert_eq!(out, "hello world");
    }

    #[cfg(unix)]
    #[test]
    fn multiline_passes_through() {
        let mut chain = PreprocessChain::spawn(&["cat".to_string()]);
        assert!(chain.is_active());
        let out = chain.process_text("hello\nworld").unwrap();
        assert_eq!(out, "hello\nworld");
    }

    // Writes a Python filter script that drops '.' lines and flushes after each output.
    // grep/sed/tr/sort all block-buffer in pipe mode on macOS — use Python with flush=True
    // (same reason cat is used for other protocol tests — see CLAUDE.md).
    #[cfg(unix)]
    fn write_dot_filter_script(suffix: &str) -> String {
        use std::os::unix::fs::PermissionsExt;
        const SCRIPT: &[u8] = b"#!/usr/bin/env python3\nimport sys\nfor line in sys.stdin:\n    s=line.rstrip('\\n')\n    if s != '.':\n        print(s, flush=True)\n";
        let path = format!(
            "{}/ir-test-{}-{}.py",
            std::env::temp_dir().display(),
            suffix,
            std::process::id()
        );
        std::fs::write(&path, SCRIPT).unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn sentinel_handles_dropped_line() {
        let path = write_dot_filter_script("filter");
        let mut handle = PreprocessHandle::spawn(&path).unwrap();
        // Punctuation-only line → filter drops it → sentinel must unblock read_line()
        let out = handle.process_line(".").unwrap();
        assert_eq!(
            out, "",
            "filtered line should return empty string, not deadlock"
        );
        let out = handle.process_line("hello").unwrap();
        assert_eq!(out, "hello");
        std::fs::remove_file(&path).ok();
    }

    #[cfg(unix)]
    #[test]
    fn sentinel_handles_mixed_document() {
        let path = write_dot_filter_script("filter2");
        let mut handle = PreprocessHandle::spawn(&path).unwrap();
        let text = "first line\n.\nsecond line\n.\nthird line";
        let out = handle.process_text(text).unwrap();
        assert_eq!(out, "first line\n\nsecond line\n\nthird line");
        std::fs::remove_file(&path).ok();
    }
}
