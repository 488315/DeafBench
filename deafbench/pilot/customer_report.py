"""Local customer report data, HTML rendering, and PDF rendering."""

from __future__ import annotations

import hashlib
import html
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import jiwer

from deafbench.metrics import evaluate_dataset, evaluate_speaker_attribution
from deafbench.parser import align_records, parse_jsonl
from deafbench.pilot.taxonomy import (
    category_label,
    category_values,
    classify_failure,
    factor_label,
    severity_label,
)


_model_labels = {
    "Qwen/Qwen3-ASR-1.7B-hf": "Qwen3-ASR 1.7B",
    "nvidia/parakeet-tdt-0.6b-v2": "Parakeet TDT 0.6B v2",
    "ibm-granite/granite-speech-4.1-2b": "Granite Speech 4.1 2B",
}
_severity_rank = {
    "no_real_impact": 0,
    "minor": 1,
    "moderate": 2,
    "major": 3,
    "critical": 4,
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _alignment(reference: str, prediction: str) -> dict[str, Any]:
    result = jiwer.process_words(
        reference if reference.strip() else " ",
        prediction if prediction.strip() else " ",
    )
    reference_words = result.references[0]
    hypothesis_words = result.hypotheses[0]
    columns: list[dict[str, str]] = []
    error_kinds: list[str] = []
    for chunk in result.alignments[0]:
        ref_words = reference_words[chunk.ref_start_idx : chunk.ref_end_idx]
        hyp_words = hypothesis_words[chunk.hyp_start_idx : chunk.hyp_end_idx]
        width = max(len(ref_words), len(hyp_words), 1)
        marker = {
            "equal": "",
            "substitute": "S",
            "delete": "D",
            "insert": "I",
        }[chunk.type]
        if chunk.type != "equal":
            error_kinds.append(chunk.type)
        for index in range(width):
            columns.append(
                {
                    "reference": ref_words[index] if index < len(ref_words) else "",
                    "hypothesis": hyp_words[index] if index < len(hyp_words) else "",
                    "kind": chunk.type,
                    "marker": marker,
                }
            )
    return {
        "columns": columns,
        "correct": result.hits,
        "deletions": result.deletions,
        "substitutions": result.substitutions,
        "insertions": result.insertions,
        "wer": float(result.wer),
        "error_kinds": tuple(dict.fromkeys(error_kinds)),
    }


def _finding_id(model_id: str, sample_id: str, problems: Sequence[Mapping[str, Any]]) -> str:
    identity = {
        "model_id": model_id,
        "sample_id": sample_id,
        "problems": [
            {
                "kind": str(problem.get("kind", "")),
                "expected": str(problem.get("expected", "")),
                "entity_type": str(problem.get("entity_type", "")),
            }
            for problem in problems
        ],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _classify_problems(
    *,
    reference_text: str,
    predicted_text: str,
    problems: list[dict[str, Any]],
    error_kinds: tuple[str, ...],
) -> dict[str, Any]:
    classifications = [
        classify_failure(
            expected=str(problem.get("expected", "")),
            reference_text=reference_text,
            predicted_text=predicted_text,
            entity_type=(
                str(problem["entity_type"])
                if problem.get("entity_type")
                else None
            ),
            finding_kind=str(problem["kind"]),
            error_kinds=error_kinds,
        )
        for problem in problems
    ]
    primary = max(classifications, key=lambda item: _severity_rank[item.severity])
    related = tuple(
        dict.fromkeys(
            factor
            for classification in classifications
            for factor in classification.related_factors
        )
    )
    return {
        "primary_category": primary.primary_category,
        "related_factors": list(related),
        "severity": primary.severity,
        "impact": primary.impact,
        "recommendation": primary.recommendation,
    }


def build_report_data(
    *,
    case_name: str,
    case_id: str,
    references_path: Path,
    prediction_paths: Sequence[Path],
    result_paths: Sequence[Path],
    reviews: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    references = parse_jsonl(str(references_path))
    references_by_id = {str(record["id"]): record for record in references}
    review_values = reviews or {}
    models: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []

    if len(prediction_paths) != len(result_paths):
        raise ValueError("Prediction and result manifest counts do not match")

    for prediction_path, result_path in zip(prediction_paths, result_paths, strict=True):
        result_manifest = _load_json(result_path)
        model = result_manifest.get("model")
        evaluations = result_manifest.get("evaluations")
        if not isinstance(model, dict) or not isinstance(evaluations, list) or len(evaluations) != 1:
            raise ValueError("Customer result manifest is incomplete")
        model_id = str(model["model_id"])
        model_label = _model_labels.get(model_id, model_id)
        evaluation = evaluations[0]
        if not isinstance(evaluation, dict) or not isinstance(evaluation.get("metrics"), dict):
            raise ValueError("Customer result metrics are incomplete")
        metrics = dict(evaluation["metrics"])
        models.append(
            {
                "model_id": model_id,
                "model_label": model_label,
                "wer_percent": float(metrics["wer_percent"]),
                "strict_recall_percent": float(metrics["strict_lexical_recall_percent"]),
                "canonical_recall_percent": float(metrics["canonical_semantic_recall_percent"]),
                "local_rtfx": float(metrics["local_rtfx"]),
                "median_latency_ms": float(metrics["median_latency_ms"]),
                "peak_vram_bytes": float(metrics["peak_vram_bytes"]),
            }
        )

        predictions = parse_jsonl(str(prediction_path))
        predictions_by_id = {str(record["id"]): record for record in predictions}
        aligned = align_records(references, predictions)
        local_metrics = evaluate_dataset(aligned)
        problems_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for failure in local_metrics.get("critical_failures", []):
            sample_id = str(failure["id"])
            reference = references_by_id[sample_id]
            critical_types = reference.get("critical_types", {})
            entity_type = (
                critical_types.get(failure["expected"])
                if isinstance(critical_types, dict)
                else None
            )
            problems_by_sample[sample_id].append(
                {
                    "kind": "critical",
                    "expected": str(failure["expected"]),
                    "entity_type": entity_type,
                }
            )
        for failure in local_metrics.get("non_speech_failures", []):
            problems_by_sample[str(failure["id"])].append(
                {
                    "kind": "sound",
                    "expected": str(failure["expected"]),
                    "entity_type": None,
                }
            )
        for item in aligned:
            reference = item["reference"]
            prediction = item["prediction"]
            if reference.get("speaker") is not None and evaluate_speaker_attribution(
                reference, prediction
            ) is False:
                sample_id = str(reference["id"])
                problems_by_sample[sample_id].append(
                    {
                        "kind": "speaker",
                        "expected": str(reference["speaker"]),
                        "entity_type": None,
                    }
                )

        for sample_id, problems in problems_by_sample.items():
            reference = references_by_id[sample_id]
            prediction = predictions_by_id[sample_id]
            reference_text = str(reference.get("text", ""))
            predicted_text = str(prediction.get("text", ""))
            alignment = _alignment(reference_text, predicted_text)
            classification = _classify_problems(
                reference_text=reference_text,
                predicted_text=predicted_text,
                problems=problems,
                error_kinds=alignment["error_kinds"],
            )
            finding_id = _finding_id(model_id, sample_id, problems)
            review = dict(review_values.get(finding_id, {}))
            customer_severity = review.get("customer_severity")
            finding = {
                "finding_id": finding_id,
                "model_id": model_id,
                "model_label": model_label,
                "sample_id": sample_id,
                "reference_text": reference_text,
                "predicted_text": predicted_text,
                "problems": problems,
                **classification,
                "alignment": {
                    key: value
                    for key, value in alignment.items()
                    if key != "error_kinds"
                },
                "review": review,
                "effective_severity": (
                    str(customer_severity)
                    if customer_severity in _severity_rank
                    else classification["severity"]
                ),
            }
            findings.append(finding)

    category_counts: dict[str, int] = defaultdict(int)
    for finding in findings:
        category_counts[str(finding["primary_category"])] += 1
    return {
        "schema_version": 1,
        "case_name": case_name,
        "case_id": case_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(references),
        "models": models,
        "findings": findings,
        "category_counts": dict(category_counts),
    }


def _alignment_html(alignment: Mapping[str, Any]) -> str:
    columns = alignment["columns"]
    cells = []
    for row_key in ("reference", "hypothesis"):
        label = "REF:" if row_key == "reference" else "HYP:"
        row = [f'<th scope="row">{label}</th>']
        for column in columns:
            kind = html.escape(str(column["kind"]))
            token = html.escape(str(column[row_key])) or "&nbsp;"
            row.append(f'<td class="token {kind}">{token}</td>')
        cells.append("<tr>" + "".join(row) + "</tr>")
    marker_row = ['<th scope="row"><span class="sr-only">Error type:</span></th>']
    for column in columns:
        marker = html.escape(str(column["marker"])) or "&nbsp;"
        marker_row.append(f'<td class="marker {html.escape(str(column["kind"]))}">{marker}</td>')
    cells.append("<tr>" + "".join(marker_row) + "</tr>")
    return '<div class="alignment-scroll"><table class="alignment"><tbody>' + "".join(cells) + "</tbody></table></div>"


def _finding_html(finding: Mapping[str, Any]) -> str:
    related = " · ".join(
        html.escape(factor_label(str(value)))
        for value in finding.get("related_factors", [])
    ) or "None"
    review = finding.get("review") if isinstance(finding.get("review"), dict) else {}
    review_lines = []
    if review:
        review_lines.append('<span class="review-status">Reviewed</span>')
        if review.get("context"):
            review_lines.append('<span class="review-status">Customer context added</span>')
        if review.get("customer_severity"):
            review_lines.append('<span class="review-status">Severity adjusted</span>')
    problems = ", ".join(
        html.escape(str(problem.get("expected", "")))
        for problem in finding.get("problems", [])
    )
    alignment = finding["alignment"]
    severity = str(finding["severity"])
    customer_severity = review.get("customer_severity")
    severity_block = f"<dd>{html.escape(severity_label(severity))}</dd>"
    if customer_severity:
        severity_block += (
            f'<dt>Customer severity</dt><dd>{html.escape(severity_label(str(customer_severity)))}</dd>'
            f'<dt>Reason</dt><dd>{html.escape(str(review.get("reason", "")))}</dd>'
        )
    if review.get("context"):
        severity_block += f'<dt>Customer context</dt><dd>{html.escape(str(review["context"]))}</dd>'
    search_value = " ".join(
        [
            str(finding["primary_category"]),
            str(finding["model_label"]),
            str(finding["sample_id"]),
            problems,
            related,
        ]
    ).casefold()
    return f"""
<article class="finding" data-category="{html.escape(str(finding['primary_category']))}" data-model="{html.escape(str(finding['model_id']))}" data-search="{html.escape(search_value)}">
  <header>
    <div>
      <p class="eyebrow">{html.escape(str(finding['model_label']))} · Sample {html.escape(str(finding['sample_id']))}</p>
      <h3>Caption failure</h3>
    </div>
    <span class="severity severity-{html.escape(str(finding['effective_severity']))}">{html.escape(severity_label(str(finding['effective_severity'])))}</span>
  </header>
  <div class="review-row">{''.join(review_lines)}</div>
  {_alignment_html(alignment)}
  <dl class="metrics">
    <div><dt>Correct words</dt><dd>{int(alignment['correct'])}</dd></div>
    <div><dt>Deletions</dt><dd>{int(alignment['deletions'])}</dd></div>
    <div><dt>Substitutions</dt><dd>{int(alignment['substitutions'])}</dd></div>
    <div><dt>Insertions</dt><dd>{int(alignment['insertions'])}</dd></div>
    <div><dt>Word error rate</dt><dd>{float(alignment['wer']):.3f}</dd></div>
  </dl>
  <dl class="details">
    <dt>Primary category</dt><dd>{html.escape(category_label(str(finding['primary_category'])))}</dd>
    <dt>Related factors</dt><dd>{related}</dd>
    <dt>DeafBench severity</dt>{severity_block}
    <dt>Failed information</dt><dd>{problems}</dd>
  </dl>
  <section><h4>Why this matters</h4><p>{html.escape(str(finding['impact']))}</p></section>
  <section><h4>Recommended investigation</h4><p>{html.escape(str(finding['recommendation']))}</p></section>
</article>
"""


def render_html(data: Mapping[str, Any]) -> str:
    models = data.get("models", [])
    findings = data.get("findings", [])
    model_rows = "".join(
        f"<tr><th scope='row'>{html.escape(str(model['model_label']))}</th>"
        f"<td>{float(model['wer_percent']):.1f}%</td>"
        f"<td>{float(model['strict_recall_percent']):.1f}%</td>"
        f"<td>{float(model['canonical_recall_percent']):.1f}%</td>"
        f"<td>{float(model['local_rtfx']):.2f}x</td></tr>"
        for model in models
    )
    model_options = "".join(
        f'<option value="{html.escape(str(model["model_id"]))}">{html.escape(str(model["model_label"]))}</option>'
        for model in models
    )
    category_sections = []
    for category in category_values():
        matching = [
            finding
            for finding in findings
            if finding.get("primary_category") == category
        ]
        if not matching:
            continue
        category_sections.append(
            f'<section class="category" data-category-section="{html.escape(category)}">'
            f'<h2>{html.escape(category_label(category))} <span class="count">{len(matching)}</span></h2>'
            + "".join(_finding_html(finding) for finding in matching)
            + "</section>"
        )
    if not findings:
        category_sections.append(
            '<section class="empty"><h2>No accessibility-critical failures detected</h2>'
            '<p>The evaluated samples did not produce a critical-information, speaker-attribution, or meaningful-sound failure under the current rules.</p></section>'
        )
    case_name = html.escape(str(data["case_name"]))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{case_name} - DeafBench audit</title>
<style>
:root {{ color-scheme: light; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #182230; background: #f5f7fa; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; line-height: 1.5; }}
main {{ width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 40px 0 80px; }}
h1, h2, h3, h4 {{ line-height: 1.2; color: #111827; }}
h1 {{ margin: 6px 0 8px; font-size: clamp(2rem, 5vw, 3rem); }}
h2 {{ margin-top: 42px; border-bottom: 1px solid #d7dde5; padding-bottom: 10px; }}
h3 {{ margin: 2px 0 0; }}
h4 {{ margin-bottom: 6px; }}
.eyebrow {{ margin: 0; color: #526173; font-size: .9rem; }}
.summary, .finding, .controls, .empty {{ background: white; border: 1px solid #d7dde5; border-radius: 10px; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
.summary {{ padding: 24px; margin-top: 24px; }}
.summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(170px,1fr)); gap: 16px; }}
.summary-grid div {{ padding: 14px; background: #f8fafc; border-radius: 8px; }}
.summary-grid strong {{ display: block; font-size: 1.7rem; }}
.table-scroll, .alignment-scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; }}
.summary-table th, .summary-table td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: right; white-space: nowrap; }}
.summary-table th:first-child {{ text-align: left; }}
.controls {{ display: grid; grid-template-columns: 1fr 1fr 2fr; gap: 12px; padding: 16px; margin-top: 24px; position: sticky; top: 8px; z-index: 10; }}
.controls label {{ font-size: .85rem; font-weight: 600; }}
.controls select, .controls input {{ width: 100%; margin-top: 4px; padding: 9px 10px; border: 1px solid #aeb8c4; border-radius: 6px; background: white; color: #111827; }}
.count {{ font-size: .82rem; font-weight: 600; background: #e8edf3; border-radius: 999px; padding: 3px 8px; vertical-align: middle; }}
.finding {{ padding: 22px; margin: 16px 0; break-inside: avoid; }}
.finding > header {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; }}
.severity {{ font-size: .85rem; font-weight: 700; border-radius: 999px; padding: 5px 10px; border: 1px solid currentColor; }}
.severity-critical {{ color: #991b1b; background: #fff1f2; }}
.severity-major {{ color: #9a3412; background: #fff7ed; }}
.severity-moderate {{ color: #854d0e; background: #fefce8; }}
.severity-minor {{ color: #1e40af; background: #eff6ff; }}
.severity-no_real_impact {{ color: #374151; background: #f3f4f6; }}
.review-row {{ display: flex; flex-wrap: wrap; gap: 7px; margin-top: 10px; }}
.review-status {{ font-size: .78rem; border: 1px solid #8b9aab; border-radius: 999px; padding: 3px 7px; color: #344054; }}
.alignment {{ width: max-content; min-width: 100%; margin: 20px 0 10px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; border-collapse: separate; border-spacing: 3px 2px; }}
.alignment th {{ text-align: right; padding-right: 8px; color: #344054; }}
.alignment td {{ text-align: center; padding: 2px 4px; border-radius: 3px; white-space: nowrap; }}
.alignment .substitute {{ color: #166534; background: #dcfce7; }}
.alignment .delete {{ color: #9f1239; background: #ffe4e6; }}
.alignment .insert {{ color: #1d4ed8; background: #dbeafe; }}
.alignment .marker {{ font-weight: 800; background: transparent; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(135px,1fr)); gap: 8px; margin: 14px 0 18px; }}
.metrics div {{ background: #f8fafc; border-radius: 6px; padding: 9px; }}
.metrics dt {{ font-size: .78rem; color: #526173; }}
.metrics dd {{ margin: 1px 0 0; font-weight: 700; }}
.details {{ display: grid; grid-template-columns: minmax(140px,190px) 1fr; gap: 7px 14px; }}
.details dt {{ font-weight: 700; }}
.details dd {{ margin: 0; }}
.sr-only {{ position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }}
[hidden] {{ display: none !important; }}
@media (max-width: 700px) {{ .controls {{ grid-template-columns: 1fr; position: static; }} .details {{ grid-template-columns: 1fr; }} }}
@media print {{ body {{ background: white; }} main {{ width: 100%; padding: 0; }} .controls {{ display: none; }} .finding, .summary {{ box-shadow: none; }} }}
</style>
</head>
<body>
<main>
  <header>
    <p class="eyebrow">DeafBench audit</p>
    <h1>{case_name}</h1>
    <p>Accessibility-critical caption evaluation. Word error rate is reported separately from consequence-based findings.</p>
  </header>
  <section class="summary" aria-labelledby="summary-title">
    <h2 id="summary-title">Summary</h2>
    <div class="summary-grid">
      <div><span>Audio samples</span><strong>{int(data['sample_count'])}</strong></div>
      <div><span>Models evaluated</span><strong>{len(models)}</strong></div>
      <div><span>Findings</span><strong>{len(findings)}</strong></div>
    </div>
    <h3>Model comparison</h3>
    <div class="table-scroll"><table class="summary-table">
      <thead><tr><th scope="col">Model</th><th scope="col">WER</th><th scope="col">Strict critical recall</th><th scope="col">Canonical critical recall</th><th scope="col">Local RTFx</th></tr></thead>
      <tbody>{model_rows}</tbody>
    </table></div>
  </section>
  <section class="controls" aria-label="Finding filters">
    <label>Category<select id="category-filter"><option value="">All categories</option>{''.join(f'<option value="{html.escape(category)}">{html.escape(category_label(category))}</option>' for category in category_values() if any(f.get('primary_category') == category for f in findings))}</select></label>
    <label>Model<select id="model-filter"><option value="">All models</option>{model_options}</select></label>
    <label>Search findings<input id="finding-search" type="search" placeholder="Search sample, model, category or failed information"></label>
  </section>
  <div id="findings">{''.join(category_sections)}</div>
</main>
<script>
(() => {{
  const category = document.getElementById('category-filter');
  const model = document.getElementById('model-filter');
  const search = document.getElementById('finding-search');
  const apply = () => {{
    const query = search.value.trim().toLowerCase();
    document.querySelectorAll('.finding').forEach((finding) => {{
      const visible = (!category.value || finding.dataset.category === category.value)
        && (!model.value || finding.dataset.model === model.value)
        && (!query || finding.dataset.search.includes(query));
      finding.hidden = !visible;
    }});
    document.querySelectorAll('[data-category-section]').forEach((section) => {{
      section.hidden = !section.querySelector('.finding:not([hidden])');
    }});
  }};
  category.addEventListener('change', apply);
  model.addEventListener('change', apply);
  search.addEventListener('input', apply);
}})();
</script>
</body>
</html>
"""


def write_pdf(data: Mapping[str, Any], destination: Path) -> None:
    try:
        from fpdf import FPDF
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'PDF support is not installed. Run: python -m pip install "deafbench[audit]"'
        ) from exc

    pdf = FPDF(format="letter")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.set_title(f"{data['case_name']} - DeafBench audit")
    pdf.set_author("DeafBench")
    pdf.set_creator("DeafBench")
    pdf.set_lang("en-US")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 9, str(data["case_name"]), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0,
        6,
        "DeafBench accessibility-critical caption audit",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)
    pdf.start_section("Summary", level=0)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, "Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0, 6, f"Audio samples: {int(data['sample_count'])}", new_x="LMARGIN", new_y="NEXT"
    )
    pdf.multi_cell(
        0,
        6,
        f"Models evaluated: {len(data.get('models', []))}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.multi_cell(
        0,
        6,
        f"Findings: {len(data.get('findings', []))}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)
    for model in data.get("models", []):
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(
            0, 6, str(model["model_label"]), new_x="LMARGIN", new_y="NEXT"
        )
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(
            0,
            5,
            f"WER {float(model['wer_percent']):.1f}% | Strict critical recall {float(model['strict_recall_percent']):.1f}% | Canonical critical recall {float(model['canonical_recall_percent']):.1f}% | Local RTFx {float(model['local_rtfx']):.2f}x",
            new_x="LMARGIN",
            new_y="NEXT",
        )
    findings = list(data.get("findings", []))
    for category in category_values():
        matching = [f for f in findings if f.get("primary_category") == category]
        if not matching:
            continue
        pdf.start_section(category_label(category), level=0)
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(
            0,
            7,
            f"{category_label(category)} ({len(matching)})",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        for finding in matching:
            pdf.start_section(
                f"{finding['model_label']} - sample {finding['sample_id']}", level=1
            )
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(
                0,
                6,
                f"{finding['model_label']} - sample {finding['sample_id']}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(
                0,
                5,
                f"Severity: {severity_label(str(finding['effective_severity']))}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.set_font("Courier", "", 8)
            pdf.multi_cell(
                0,
                4.5,
                f"REF: {finding['reference_text']}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.multi_cell(
                0,
                4.5,
                f"HYP: {finding['predicted_text']}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            alignment = finding["alignment"]
            pdf.multi_cell(
                0,
                4.5,
                f"Correct words: {alignment['correct']} | Deletions: {alignment['deletions']} | Substitutions: {alignment['substitutions']} | Insertions: {alignment['insertions']} | WER: {float(alignment['wer']):.3f}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(
                0,
                5,
                f"Primary category: {category_label(str(finding['primary_category']))}",
                new_x="LMARGIN",
                new_y="NEXT",
            )
            factors = ", ".join(factor_label(str(value)) for value in finding.get("related_factors", [])) or "None"
            pdf.multi_cell(
                0, 5, f"Related factors: {factors}", new_x="LMARGIN", new_y="NEXT"
            )
            review = finding.get("review") if isinstance(finding.get("review"), dict) else {}
            if review.get("customer_severity"):
                pdf.multi_cell(
                    0,
                    5,
                    f"DeafBench severity: {severity_label(str(finding['severity']))}",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
                pdf.multi_cell(
                    0,
                    5,
                    f"Customer severity: {severity_label(str(review['customer_severity']))}",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
                pdf.multi_cell(
                    0,
                    5,
                    f"Reason: {review.get('reason', '')}",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
            if review.get("context"):
                pdf.multi_cell(
                    0,
                    5,
                    f"Customer context: {review['context']}",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )
            pdf.set_font("Helvetica", "B", 9)
            pdf.multi_cell(
                0, 5, "Why this matters", new_x="LMARGIN", new_y="NEXT"
            )
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(
                0, 5, str(finding["impact"]), new_x="LMARGIN", new_y="NEXT"
            )
            pdf.set_font("Helvetica", "B", 9)
            pdf.multi_cell(
                0, 5, "Recommended investigation", new_x="LMARGIN", new_y="NEXT"
            )
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(
                0,
                5,
                str(finding["recommendation"]),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.ln(3)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(destination))


def write_reports(data: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    html_path = destination / "index.html"
    pdf_path = destination / "report.pdf"
    html_path.write_text(render_html(data), encoding="utf-8", newline="\n")
    write_pdf(data, pdf_path)
    return html_path, pdf_path
