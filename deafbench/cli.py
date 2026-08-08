import argparse
import sys
from typing import Optional, List
from .parser import parse_jsonl, align_records
from .metrics import evaluate_dataset
from .report import generate_markdown_report

def format_terminal_output(metrics: dict) -> str:
    lines = [
        "DeafBench v0.1",
        "",
        f"Samples: {metrics['samples']}",
        "",
        f"WER                       {metrics['wer']:>6.1f}%",
        f"Critical Information      {metrics['critical_recall']:>6.1f}%",
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

def main(args: Optional[List[str]] = None) -> None:
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
    
    parsed = parser.parse_args(args)
    
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

if __name__ == "__main__":
    main()
