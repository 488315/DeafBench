import sys
import subprocess
from types import ModuleType, SimpleNamespace

import pytest

from deafbench.leaderboard.zipformer import PinnedZipformerContract
from deafbench.leaderboard import zipformer_runner
from deafbench.leaderboard.zipformer_runner import _runner_argv


def test_zipformer_contract_pins_public_dataset_and_model_revisions():
    calls = []

    def load_dataset(*args, **kwargs):
        calls.append(("dataset", args, kwargs))
        return "dataset"

    def snapshot_download(*args, **kwargs):
        calls.append(("model", args, kwargs))
        return "snapshot"

    contract = PinnedZipformerContract()
    arguments = SimpleNamespace(
        dataset_path=contract.dataset_id,
        dataset="librispeech",
        split="test.clean",
        streaming=False,
    )

    assert contract.load_dataset(load_dataset, arguments) == "dataset"
    assert contract.snapshot_model(
        snapshot_download,
        contract.model_id,
        allow_patterns=("*.pt", "*.model", "*.yaml"),
    ) == "snapshot"
    assert calls == [
        (
            "dataset",
            (contract.dataset_id, "librispeech"),
            {
                "revision": contract.dataset_revision,
                "split": "test.clean",
                "streaming": False,
                "token": False,
            },
        ),
        (
            "model",
            (contract.model_id,),
            {
                "allow_patterns": ("*.pt", "*.model", "*.yaml"),
                "revision": contract.model_revision,
            },
        ),
    ]


@pytest.mark.parametrize(
    ("dataset", "split"),
    (("librispeech", "validation"), ("common_voice", "test")),
)
def test_zipformer_contract_rejects_nonofficial_dataset_splits(dataset, split):
    contract = PinnedZipformerContract()
    arguments = SimpleNamespace(
        dataset_path=contract.dataset_id,
        dataset=dataset,
        split=split,
        streaming=False,
    )

    with pytest.raises(ValueError, match="unsupported official dataset/split"):
        contract.load_dataset(lambda *args, **kwargs: None, arguments)


def test_zipformer_runner_translates_pinned_nonstreaming_arguments():
    contract = PinnedZipformerContract()
    arguments = SimpleNamespace(
        dataset="librispeech",
        split="test.clean",
        device=0,
        batch_size=32,
        warmup_steps=1,
        max_eval_samples=64,
        streaming=False,
    )

    assert _runner_argv(arguments, contract) == [
        "--model_id",
        contract.model_id,
        "--dataset_path",
        contract.dataset_id,
        "--dataset",
        "librispeech",
        "--split",
        "test.clean",
        "--device",
        "0",
        "--batch_size",
        "32",
        "--warmup_steps",
        "1",
        "--max_eval_samples",
        "64",
        "--no-streaming",
    ]


def test_zipformer_runner_rejects_cpu_execution():
    arguments = SimpleNamespace(device=-1)

    with pytest.raises(ValueError, match="requires a CUDA device"):
        _runner_argv(arguments, PinnedZipformerContract())


def test_zipformer_runner_rejects_nonofficial_dataset_split():
    arguments = SimpleNamespace(dataset="librispeech", split="train", device=0)

    with pytest.raises(ValueError, match="unsupported official dataset/split"):
        _runner_argv(arguments, PinnedZipformerContract())


def test_zipformer_runner_rejects_preloaded_upstream_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "run_eval", ModuleType("run_eval"))

    with pytest.raises(RuntimeError, match="already loaded: run_eval"):
        zipformer_runner._require_fresh_pinned_imports()


def test_zipformer_runner_accepts_clean_pinned_source(tmp_path, monkeypatch):
    checkout = tmp_path / "source"
    required = checkout / "nested" / "runner.py"
    required.parent.mkdir(parents=True)
    required.write_text("", encoding="utf-8")
    responses = iter(
        (
            subprocess.CompletedProcess([], 0, stdout="abc123\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        )
    )
    monkeypatch.setattr(
        zipformer_runner.subprocess,
        "run",
        lambda *args, **kwargs: next(responses),
    )

    zipformer_runner._require_source(
        checkout,
        "abc123",
        "nested/runner.py",
        "runner",
    )


def test_zipformer_runner_rejects_modified_source(tmp_path, monkeypatch):
    checkout = tmp_path / "source"
    required = checkout / "runner.py"
    checkout.mkdir()
    required.write_text("", encoding="utf-8")
    responses = iter(
        (
            subprocess.CompletedProcess([], 0, stdout="abc123\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout=" M runner.py\n", stderr=""),
        )
    )
    monkeypatch.setattr(
        zipformer_runner.subprocess,
        "run",
        lambda *args, **kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match="source is modified"):
        zipformer_runner._require_source(
            checkout,
            "abc123",
            "runner.py",
            "runner",
        )


def test_zipformer_runner_rejects_missing_source_file(tmp_path):
    with pytest.raises(RuntimeError, match="is missing runner.py"):
        zipformer_runner._require_source(
            tmp_path,
            "abc123",
            "runner.py",
            "runner",
        )


def test_zipformer_runner_rejects_wrong_revision(tmp_path, monkeypatch):
    required = tmp_path / "runner.py"
    required.write_text("", encoding="utf-8")
    responses = iter(
        (
            subprocess.CompletedProcess([], 0, stdout="wrong\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        )
    )
    monkeypatch.setattr(
        zipformer_runner.subprocess,
        "run",
        lambda *args, **kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match="revision mismatch"):
        zipformer_runner._require_source(
            tmp_path,
            "abc123",
            "runner.py",
            "runner",
        )


def test_zipformer_runner_executes_reviewed_upstream_contract(
    tmp_path,
    monkeypatch,
):
    captured = {}
    runner_module = SimpleNamespace(
        data_utils=SimpleNamespace(load_data=None),
        snapshot_download=None,
        get_parser=lambda: SimpleNamespace(
            parse_args=lambda argv: captured.setdefault("argv", argv)
        ),
        main=lambda parsed: captured.setdefault("parsed", parsed),
    )
    cuda = SimpleNamespace(
        set_device=lambda device: captured.setdefault("device", device),
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 1234,
    )
    torch_module = ModuleType("torch")
    torch_module.cuda = cuda
    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: None
    hub_module = ModuleType("huggingface_hub")
    hub_module.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setattr(zipformer_runner, "_require_source", lambda *args: None)
    monkeypatch.setattr(
        zipformer_runner,
        "open_asr_evaluator",
        lambda path: SimpleNamespace(validate=lambda: None),
    )
    monkeypatch.setattr(
        zipformer_runner.importlib,
        "import_module",
        lambda name: runner_module,
    )
    monkeypatch.setattr(
        zipformer_runner,
        "_require_fresh_pinned_imports",
        lambda: None,
    )
    monkeypatch.setattr(
        zipformer_runner,
        "verify_evaluation_policy",
        lambda path: captured.setdefault("policy", path),
    )
    clock = iter((10.0, 12.3456))
    monkeypatch.setattr(zipformer_runner.time, "time", lambda: next(clock))
    arguments = SimpleNamespace(
        runner_repo=tmp_path / "runner",
        official_repo=tmp_path / "official",
        icefall_repo=tmp_path / "icefall",
        evaluation_policy=tmp_path / "evaluation-policy.json",
        output_dir=tmp_path / "output",
        dataset="librispeech",
        split="test.clean",
        device=0,
        batch_size=32,
        warmup_steps=1,
        max_eval_samples=2,
        streaming=False,
    )

    previous_cwd = zipformer_runner.os.getcwd()
    previous_path = list(sys.path)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    try:
        summary = zipformer_runner.run(arguments)
        restored_state = (
            zipformer_runner.os.getcwd(),
            list(sys.path),
            sys.dont_write_bytecode,
        )
    finally:
        zipformer_runner.os.chdir(previous_cwd)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_dont_write_bytecode

    assert summary == {"wall_seconds": 2.346, "peak_vram_bytes": 1234}
    assert restored_state == (
        previous_cwd,
        previous_path,
        previous_dont_write_bytecode,
    )
    assert captured["device"] == 0
    assert captured["policy"] == tmp_path / "evaluation-policy.json"
    assert captured["parsed"] == captured["argv"]
    assert runner_module.data_utils.load_data is not None
    assert runner_module.snapshot_download is not None


def test_zipformer_runner_restores_process_state_after_failure(
    tmp_path,
    monkeypatch,
):
    runner_module = SimpleNamespace(
        data_utils=SimpleNamespace(load_data=None),
        snapshot_download=None,
        get_parser=lambda: SimpleNamespace(parse_args=lambda argv: argv),
        main=lambda parsed: (_ for _ in ()).throw(RuntimeError("runner failed")),
    )
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(
        set_device=lambda device: None,
        reset_peak_memory_stats=lambda: None,
    )
    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = lambda *args, **kwargs: None
    hub_module = ModuleType("huggingface_hub")
    hub_module.snapshot_download = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setattr(zipformer_runner, "_require_source", lambda *args: None)
    monkeypatch.setattr(
        zipformer_runner,
        "open_asr_evaluator",
        lambda path: SimpleNamespace(validate=lambda: None),
    )
    monkeypatch.setattr(
        zipformer_runner,
        "_require_fresh_pinned_imports",
        lambda: None,
    )
    monkeypatch.setattr(
        zipformer_runner,
        "verify_evaluation_policy",
        lambda path: None,
    )
    monkeypatch.setattr(
        zipformer_runner.importlib,
        "import_module",
        lambda name: runner_module,
    )
    arguments = SimpleNamespace(
        runner_repo=tmp_path / "runner",
        official_repo=tmp_path / "official",
        icefall_repo=tmp_path / "icefall",
        evaluation_policy=tmp_path / "evaluation-policy.json",
        output_dir=tmp_path / "output",
        dataset="librispeech",
        split="test.clean",
        device=0,
        batch_size=32,
        warmup_steps=1,
        max_eval_samples=2,
        streaming=False,
    )
    previous_cwd = zipformer_runner.os.getcwd()
    previous_path = list(sys.path)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    try:
        with pytest.raises(RuntimeError, match="runner failed"):
            zipformer_runner.run(arguments)
        restored_state = (
            zipformer_runner.os.getcwd(),
            list(sys.path),
            sys.dont_write_bytecode,
        )
    finally:
        zipformer_runner.os.chdir(previous_cwd)
        sys.path[:] = previous_path
        sys.dont_write_bytecode = previous_dont_write_bytecode

    assert restored_state == (
        previous_cwd,
        previous_path,
        previous_dont_write_bytecode,
    )


def test_zipformer_runner_main_parses_cli_arguments(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        zipformer_runner,
        "run",
        lambda args: captured.setdefault("args", args),
    )

    exit_code = zipformer_runner.main(
        [
            "--runner-repo",
            str(tmp_path / "runner"),
            "--official-repo",
            str(tmp_path / "official"),
            "--icefall-repo",
            str(tmp_path / "icefall"),
            "--evaluation-policy",
            str(tmp_path / "evaluation-policy.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--dataset",
            "librispeech",
            "--split",
            "test.clean",
            "--streaming",
        ]
    )

    assert exit_code == 0
    assert captured["args"].streaming is True
    assert captured["args"].batch_size == 16
    assert captured["args"].evaluation_policy == str(
        tmp_path / "evaluation-policy.json"
    )
