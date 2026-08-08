# DeafBench Evaluation Report

- **Reference File:** `benchmarks\non-speech-v1\references.jsonl`
- **Prediction File:** `.\benchmarks\core-v1\model-a.jsonl`
- **Total Samples:** 12

## Summary Metrics

| Metric | Value |
| --- | --- |
| **Word Error Rate (WER)** | 100.0% |
| **Critical Information Recall** | 0.0% (0/20) |
| **Non-Speech Information Recall** | 0.0% (0/19) |
| **Speaker Attribution Accuracy** | N/A |
| **Median Latency** | N/A |

## Critical Information Failures

Detected **20** critical information failures:

| Sample ID | Missing Critical Term | Output Text |
| --- | --- | --- |
| `ns-001` | **remain seated** | ** |
| `ns-002` | **Taylor** | ** |
| `ns-002` | **front desk** | ** |
| `ns-003` | **support desk** | ** |
| `ns-003` | **after the meeting** | ** |
| `ns-004` | **outside the office** | ** |
| `ns-004` | **Morgan** | ** |
| `ns-005` | **security update** | ** |
| `ns-006` | **nearest safe exit** | ** |
| `ns-007` | **Jordan** | ** |
| `ns-007` | **blue chair** | ** |
| `ns-008` | **meeting will continue** | ** |
| `ns-009` | **backup completed successfully** | ** |
| `ns-009` | **midnight** | ** |
| `ns-010` | **Casey** | ** |
| `ns-010` | **second floor** | ** |
| `ns-011` | **front entrance** | ** |
| `ns-011` | **before leaving** | ** |
| `ns-012` | **emergency instructions** | ** |
| `ns-012` | **reception desk** | ** |

## Non-Speech Information Failures

Detected **19** non-speech information failures:

| Sample ID | Missing Sound Event | Output Text |
| --- | --- | --- |
| `ns-001` | **[alarm]** | ** |
| `ns-002` | **[door closes]** | ** |
| `ns-003` | **[phone rings]** | ** |
| `ns-004` | **[knock]** | ** |
| `ns-005` | **[error notification]** | ** |
| `ns-006` | **[siren]** | ** |
| `ns-007` | **[alarm]** | ** |
| `ns-007` | **[door closes]** | ** |
| `ns-008` | **[phone rings]** | ** |
| `ns-008` | **[knock]** | ** |
| `ns-009` | **[error notification]** | ** |
| `ns-009` | **[alarm]** | ** |
| `ns-010` | **[siren]** | ** |
| `ns-010` | **[phone rings]** | ** |
| `ns-011` | **[knock]** | ** |
| `ns-011` | **[door closes]** | ** |
| `ns-012` | **[alarm]** | ** |
| `ns-012` | **[phone rings]** | ** |
| `ns-012` | **[knock]** | ** |
