use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::{Builder, TempDir};

const INHERITED_MODEL_ENV_VARS: &[&str] = &[
    "IR_COMBINED_MODEL",
    "IR_QWEN_MODEL",
    "IR_EXPANDER_MODEL",
    "IR_RERANKER_MODEL",
    "QMD_EMBEDDING_MODEL",
    "QMD_EMBED_MODEL",
    "QMD_EXPANDER_MODEL",
    "QMD_EXPAND_MODEL",
    "QMD_RERANKER_MODEL",
    "QMD_RERANK_MODEL",
    "QMD_MODEL_DIRS",
];

struct Fixture {
    _root: TempDir,
    config_dir: PathBuf,
    collection_dir: PathBuf,
    empty_model_dir: PathBuf,
    invalid_model: PathBuf,
}

impl Fixture {
    fn registered() -> Self {
        let tmp_root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".tmp");
        fs::create_dir_all(&tmp_root).expect("create project-local temporary directory");
        let root = Builder::new()
            .prefix("sync-cli-")
            .tempdir_in(tmp_root)
            .expect("create unique CLI integration fixture");
        let config_dir = root.path().join("config");
        let collection_dir = root.path().join("empty-collection");
        let empty_model_dir = root.path().join("empty-models");
        fs::create_dir_all(&collection_dir).expect("create empty collection fixture");
        fs::create_dir_all(&empty_model_dir).expect("create empty model directory");

        let fixture = Self {
            invalid_model: root.path().join("must-not-be-loaded.gguf"),
            _root: root,
            config_dir,
            collection_dir,
            empty_model_dir,
        };

        let mut add = fixture.ir();
        add.args(["collection", "add", "empty"])
            .arg(&fixture.collection_dir);
        assert_eq!(
            snapshot(run(add)),
            (
                true,
                "added collection 'empty'\n".to_string(),
                String::new(),
            )
        );

        fixture
    }

    fn ir(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ir"));
        for key in INHERITED_MODEL_ENV_VARS {
            command.env_remove(key);
        }
        command
            .env("IR_CONFIG_DIR", &self.config_dir)
            .env("IR_MODEL_DIRS", &self.empty_model_dir)
            .env("IR_EMBEDDING_MODEL", &self.invalid_model);
        command
    }
}

fn run(mut command: Command) -> Output {
    command.output().expect("run compiled ir binary")
}

fn snapshot(output: Output) -> (bool, String, String) {
    (
        output.status.success(),
        String::from_utf8(output.stdout).expect("stdout should be UTF-8"),
        String::from_utf8(output.stderr).expect("stderr should be UTF-8"),
    )
}

fn assert_model_failure_after_update(output: Output, expected_added: usize) {
    let (success, stdout, stderr) = snapshot(output);
    let expected_stdout =
        format!("updating 'empty'…\n  {expected_added} added, 0 updated, 0 deactivated\n");
    assert_eq!(
        (
            success,
            stdout,
            stderr.contains("IR_EMBEDDING_MODEL="),
            stderr.contains("is not a file, directory, or known HuggingFace repo ID"),
        ),
        (false, expected_stdout, true, true)
    );
}

#[test]
fn empty_collection_sync_and_embed_update_before_skipping_model_loading() {
    let fixture = Fixture::registered();
    let expected_phases = concat!(
        "updating 'empty'…\n",
        "  0 added, 0 updated, 0 deactivated\n",
        "embedding 'empty'…\n",
        "  0 documents, 0 chunks embedded\n",
    );

    let mut sync = fixture.ir();
    sync.args(["sync", "empty"]);
    assert_eq!(
        snapshot(run(sync)),
        (true, expected_phases.to_string(), String::new())
    );

    let mut embed = fixture.ir();
    embed.args(["embed", "empty"]);
    assert_eq!(
        snapshot(run(embed)),
        (true, expected_phases.to_string(), String::new())
    );
}

#[test]
fn nonempty_sync_and_embed_commit_updates_before_model_loading_failure() {
    let fixture = Fixture::registered();

    fs::write(fixture.collection_dir.join("first.md"), "# First\n").expect("create first document");
    let mut first_sync = fixture.ir();
    first_sync.args(["sync", "empty"]);
    assert_model_failure_after_update(run(first_sync), 1);

    let mut repeated_sync = fixture.ir();
    repeated_sync.args(["sync", "empty"]);
    assert_model_failure_after_update(run(repeated_sync), 0);

    fs::write(fixture.collection_dir.join("second.md"), "# Second\n")
        .expect("create second document");
    let mut first_embed = fixture.ir();
    first_embed.args(["embed", "empty"]);
    assert_model_failure_after_update(run(first_embed), 1);

    let mut repeated_embed = fixture.ir();
    repeated_embed.args(["embed", "empty"]);
    assert_model_failure_after_update(run(repeated_embed), 0);
}
