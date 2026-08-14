import argparse
import sys
from typing import Optional, List
from .parser import parse_jsonl, align_records
from .metrics import evaluate_dataset
from .report import generate_markdown_report
from .benchmark.models import MODEL_NAMES
from .leaderboard.cli import add_leaderboard_parser, run_leaderboard


_BENCHMARK_RUNTIME_UNAVAILABLE = (
    "Benchmark runtime is not available in this build; "
    "use `deafbench compare` until a release with benchmark runner support is "
    "installed."
)


def format_terminal_output(metrics: dict) -> str:
    canonical_recall = metrics.get(
        "canonical_critical_recall", metrics["critical_recall"]
    )
    strict_recall = metrics.get("strict_critical_recall", canonical_recall)
    lines = [
        "DeafBench v0.1",
        "",
        f"Samples: {metrics['samples']}",
        "",
        f"Orthographic WER          {metrics.get('orthographic_wer', metrics['wer']):>6.1f}%",
        f"Normalized WER            {metrics.get('normalized_wer', metrics['wer']):>6.1f}%",
        f"Orthographic CER          {metrics.get('orthographic_cer', metrics.get('cer', 0.0)):>6.1f}%",
        f"Normalized CER            {metrics.get('normalized_cer', metrics.get('cer', 0.0)):>6.1f}%",
        f"Normalization policy      {metrics.get('normalization_policy', 'legacy-unspecified')}",
        f"Strict Critical Information    {strict_recall:>6.1f}%",
        f"Canonical Critical Information {canonical_recall:>6.1f}%",
        f"WER edits (S/I/D)         {metrics.get('substitutions', 0)}/"
        f"{metrics.get('insertions', 0)}/{metrics.get('deletions', 0)}",
    ]

    if metrics.get("non_speech_recall") is not None:
        lines.append(f"Non-Speech Information    {metrics['non_speech_recall']:>6.1f}%")
    else:
        lines.append(f"Non-Speech Information    {'N/A':>6}")

    if metrics.get("speaker_accuracy") is not None:
        lines.append(f"Speaker Attribution       {metrics['speaker_accuracy']:>6.1f}%")
        
    if metrics.get("median_latency_ms") is not None:
        latency_sec = metrics['median_latency_ms'] / 1000.0
        lines.append(f"Median Latency            {latency_sec:>6.1f}s")
        
    failures = metrics.get("critical_failures", [])
    lines.append("")
    if failures:
        lines.append(f"[!] {len(failures)} critical-information failure{'s' if len(failures) > 1 else ''} detected")
    else:
        lines.append("[+] 0 critical-information failures detected")
        
    return "\n".join(lines)


def _run_recorder(recorder_args: list[str]) -> int:
    """Lazy-load and run the optional recorder runtime."""
    try:
        from .recorder.app import main as recorder_main
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise SystemExit(
                'Recorder dependencies are not installed. Run: python -m pip install "deafbench[recorder]"'
            ) from exc
        raise
    return recorder_main(recorder_args)


def _run_benchmark(benchmark_args: list[str]) -> int:
    """Lazy-load and run the optional benchmark orchestration."""
    try:
        from .benchmark.runner import main as benchmark_main
    except ModuleNotFoundError as exc:
        if exc.name == "deafbench.benchmark.runner":
            raise SystemExit(_BENCHMARK_RUNTIME_UNAVAILABLE) from exc
        raise
    return benchmark_main(benchmark_args)


def _run_audit(audit_args: list[str]) -> int:
    """Lazy-load and run the customer-local audit workflow."""
    from .pilot.cli import main as audit_main

    return audit_main(audit_args, prog="deafbench audit")


def _run_dev_corpus(dev_corpus_args: list[str]) -> int:
    """Lazy-load the public development corpus workflow."""
    from .leaderboard.dev_cli import main as dev_corpus_main

    return dev_corpus_main(dev_corpus_args)


def _run_stress(stress_args: list[str]) -> int:
    """Lazy-load the local accessibility stress workflow."""
    from .benchmark.stress_cli import main as stress_main

    return stress_main(stress_args)


def _build_recorder_args(parsed: argparse.Namespace) -> list[str]:
    recorder_args = ["--dataset", parsed.dataset]
    for option, value in (
        ("--repo-root", parsed.repo_root),
        ("--references", parsed.recorder_references),
        ("--audio-dir", parsed.audio_dir),
    ):
        if value is not None:
            recorder_args.extend([option, value])
    return recorder_args


def main(args: Optional[List[str]] = None) -> int | None:
    arguments = list(args) if args is not None else sys.argv[1:]
    if arguments[:1] == ["audit"]:
        return _run_audit(arguments[1:])
    if arguments[:1] == ["dev-corpus"]:
        return _run_dev_corpus(arguments[1:])
    if arguments[:1] == ["stress"]:
        return _run_stress(arguments[1:])

    parser = argparse.ArgumentParser(
        prog="deafbench",
        description="DeafBench: Accessibility-focused evaluation for AI captions and ASR systems."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare reference captions with model predictions."
    )
    compare_parser.add_argument("references", help="Path to reference JSONL file")
    compare_parser.add_argument("predictions", help="Path to predictions JSONL file")
    
    # report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate a Markdown evaluation report."
    )
    report_parser.add_argument("references", help="Path to reference JSONL file")
    report_parser.add_argument("predictions", help="Path to predictions JSONL file")
    report_parser.add_argument("-o", "--output", help="Output markdown file path", default="report.md")

    # recorder command
    recorder_parser = subparsers.add_parser(
        "recorder",
        help="Launch the DeafBench dataset recorder."
    )
    recorder_parser.add_argument("--dataset", default="core-v1", help="Benchmark directory under benchmarks/")
    recorder_parser.add_argument("--repo-root", help="Workspace root for benchmark files")
    recorder_parser.add_argument("--references", dest="recorder_references", help="Reference JSONL override")
    recorder_parser.add_argument("--audio-dir", help="Recorded WAV output directory")

    # benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run a complete DeafBench model benchmark.",
    )
    benchmark_parser.add_argument(
        "dataset",
        help="Benchmark directory under benchmarks/",
    )
    benchmark_parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_NAMES,
    )
    benchmark_parser.add_argument(
        "--audio-source",
        choices=("auto", "human", "synthetic"),
        default="auto",
    )
    benchmark_parser.add_argument("--repo-root", dest="benchmark_repo_root")
    benchmark_parser.add_argument("--scene-profile", default="default-v1")
    benchmark_parser.add_argument("--seed", type=int, default=42)

    subparsers.add_parser(
        "audit",
        help="Run or verify a customer-local zero-custody audit.",
    )

    subparsers.add_parser(
        "dev-corpus",
        help="Materialize the pinned public development cohort.",
    )

    subparsers.add_parser(
        "stress",
        help="Run the local accessibility stress benchmark.",
    )

    add_leaderboard_parser(subparsers)
    
    parsed = parser.parse_args(arguments)

    if parsed.command == "recorder":
        return _run_recorder(_build_recorder_args(parsed))

    if parsed.command == "benchmark":
        benchmark_args = [
            parsed.dataset,
            "--model", parsed.model,
            "--audio-source", parsed.audio_source,
            "--scene-profile", parsed.scene_profile,
            "--seed", str(parsed.seed),
        ]
        if parsed.benchmark_repo_root is not None:
            benchmark_args.extend(["--repo-root", parsed.benchmark_repo_root])
        return _run_benchmark(benchmark_args)

    if parsed.command == "leaderboard":
        return run_leaderboard(parsed)
    
    try:
        references = parse_jsonl(parsed.references)
        predictions = parse_jsonl(parsed.predictions)
    except Exception as e:
        print(f"Error reading input files: {e}", file=sys.stderr)
        sys.exit(1)
        
    aligned = align_records(references, predictions)
    try:
        metrics = evaluate_dataset(aligned)
    except ValueError as e:
        print(f"Error evaluating dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    if parsed.command == "compare":
        print(format_terminal_output(metrics))
    elif parsed.command == "report":
        md = generate_markdown_report(metrics, parsed.references, parsed.predictions)
        try:
            with open(parsed.output, "w", encoding="utf-8") as f:
                f.write(md)
        except OSError as e:
            print(f"Error writing report: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Report successfully saved to {parsed.output}")

    return None


if __name__ == "__main__":
    raise SystemExit(main())
