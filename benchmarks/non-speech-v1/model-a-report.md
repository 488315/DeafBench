# DeafBench Evaluation Report

- **Reference File:** `benchmarks\non-speech-v1\references.jsonl`
- **Prediction File:** `benchmarks\non-speech-v1\model-a.jsonl`
- **Total Samples:** 12

## Summary Metrics

| Metric | Value |
| --- | --- |
| **Word Error Rate (WER)** | 2.0% |
| **Critical Information Recall** | 95.0% (19/20) |
| **Non-Speech Information Recall** | 0.0% (0/19) |
| **Speaker Attribution Accuracy** | N/A |
| **Median Latency** | N/A |

## Critical Information Failures

Detected **1** critical information failure:

| Sample ID | Missing Critical Term | Output Text |
| --- | --- | --- |
| `ns-006` | **nearest safe exit** | *Leave the area using the nearest safe access.* |

## Non-Speech Information Failures

Detected **19** non-speech information failures:

| Sample ID | Missing Sound Event | Output Text |
| --- | --- | --- |
| `ns-001` | **[alarm]** | *Please remain seated until the announcement is finished.* |
| `ns-002` | **[door closes]** | *Taylor is waiting in the hallway near the front desk.* |
| `ns-003` | **[phone rings]** | *The support desk will call you back after the meeting.* |
| `ns-004` | **[knock]** | *Someone is waiting outside the office for Morgan.* |
| `ns-005` | **[error notification]** | *The application finished installing the security update.* |
| `ns-006` | **[siren]** | *Leave the area using the nearest safe access.* |
| `ns-007` | **[alarm]** | *Jordan placed the package beside the blue chair.* |
| `ns-007` | **[door closes]** | *Jordan placed the package beside the blue chair.* |
| `ns-008` | **[phone rings]** | *The meeting will continue after this short interruption.* |
| `ns-008` | **[knock]** | *The meeting will continue after this short interruption.* |
| `ns-009` | **[error notification]** | *The database backup completed successfully at midnight.* |
| `ns-009` | **[alarm]** | *The database backup completed successfully at midnight.* |
| `ns-010` | **[siren]** | *Casey should wait by the elevator on the second floor.* |
| `ns-010` | **[phone rings]** | *Casey should wait by the elevator on the second floor.* |
| `ns-011` | **[knock]** | *Check the front entrance before leaving the building.* |
| `ns-011` | **[door closes]** | *Check the front entrance before leaving the building.* |
| `ns-012` | **[alarm]** | *The emergency instructions are posted beside the reception desk.* |
| `ns-012` | **[phone rings]** | *The emergency instructions are posted beside the reception desk.* |
| `ns-012` | **[knock]** | *The emergency instructions are posted beside the reception desk.* |
