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

Detected **1** critical information failures:

| Sample ID | Missing Critical Term | Output Text |
| --- | --- | --- |
| `ns-006` | **nearest safe exit** | * Leave the area using the nearest safe access.* |
