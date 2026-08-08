from typing import Dict, Any


def _escape_markdown_table_cell(value: Any) -> str:
    """Escape user-provided content for safe Markdown table cells."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\r", "<br>")
        .replace("\n", "<br>")
    )


def generate_markdown_report(metrics: Dict[str, Any], ref_file: str, pred_file: str) -> str:
    """Generate Markdown report from metrics output."""
    lines = [
        "# DeafBench Evaluation Report",
        "",
        f"- **Reference File:** `{ref_file}`",
        f"- **Prediction File:** `{pred_file}`",
        f"- **Total Samples:** {metrics['samples']}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| **Word Error Rate (WER)** | {metrics['wer']:.1f}% |",
        f"| **Critical Information Recall** | {metrics['critical_recall']:.1f}% ({metrics['matched_critical']}/{metrics['total_critical']}) |",
    ]

    if metrics.get("non_speech_recall") is not None:
        lines.append(
            f"| **Non-Speech Information Recall** | {metrics['non_speech_recall']:.1f}% "
            f"({metrics['matched_sounds']}/{metrics['total_sounds']}) |"
        )
    else:
        lines.append("| **Non-Speech Information Recall** | N/A |")

    if metrics.get("speaker_accuracy") is not None:
        lines.append(f"| **Speaker Attribution Accuracy** | {metrics['speaker_accuracy']:>6.1f}% |")
    else:
        lines.append("| **Speaker Attribution Accuracy** | N/A |")
        
    if metrics.get("median_latency_ms") is not None:
        latency_sec = metrics["median_latency_ms"] / 1000.0
        lines.append(f"| **Median Latency** | {latency_sec:.2f}s ({metrics['median_latency_ms']:.0f} ms) |")
    else:
        lines.append("| **Median Latency** | N/A |")
        
    lines.extend([
        "",
        "## Critical Information Failures",
        ""
    ])
    
    failures = metrics.get("critical_failures", [])
    if not failures:
        lines.append("No critical information failures detected! 🎉")
    else:
        noun = "failure" if len(failures) == 1 else "failures"
        lines.append(f"Detected **{len(failures)}** critical information {noun}:")
        lines.append("")
        lines.append("| Sample ID | Missing Critical Term | Output Text |")
        lines.append("| --- | --- | --- |")
        for fail in failures:
            expected = _escape_markdown_table_cell(fail["expected"])
            predicted_text = _escape_markdown_table_cell(
                str(fail["predicted_text"]).strip()
            )
            lines.append(f"| `{fail['id']}` | **{expected}** | *{predicted_text}* |")

    if metrics.get("total_sounds", 0) > 0:
        lines.extend([
            "",
            "## Non-Speech Information Failures",
            "",
        ])
        sound_failures = metrics.get("non_speech_failures", [])
        if not sound_failures:
            lines.append("No non-speech information failures detected! 🎉")
        else:
            noun = "failure" if len(sound_failures) == 1 else "failures"
            lines.append(
                f"Detected **{len(sound_failures)}** non-speech information {noun}:"
            )
            lines.append("")
            lines.append("| Sample ID | Missing Sound Event | Output Text |")
            lines.append("| --- | --- | --- |")
            for fail in sound_failures:
                sample_id = _escape_markdown_table_cell(fail["id"])
                expected = _escape_markdown_table_cell(fail["expected"])
                predicted_text = _escape_markdown_table_cell(
                    str(fail["predicted_text"]).strip()
                )
                lines.append(f"| `{sample_id}` | **{expected}** | *{predicted_text}* |")

    lines.append("")
    return "\n".join(lines)
