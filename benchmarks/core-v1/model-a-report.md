# DeafBench Evaluation Report

- **Reference File:** `benchmarks\core-v1\references.jsonl`
- **Prediction File:** `benchmarks\core-v1\model-a.jsonl`
- **Total Samples:** 25

## Summary Metrics

| Metric | Value |
| --- | --- |
| **Word Error Rate (WER)** | 23.4% |
| **Critical Information Recall** | 88.7% (55/62) |
| **Non-Speech Information Recall** | N/A |
| **Speaker Attribution Accuracy** | N/A |
| **Median Latency** | N/A |

## Critical Information Failures

Detected **7** critical information failures:

| Sample ID | Missing Critical Term | Output Text |
| --- | --- | --- |
| `core-001` | **2:15 PM** | *My appointment with Dr. Martinez is Friday at 12.15 p.m.* |
| `core-010` | **authentication token expired** | *The API returned error code 503 because the application token expired.* |
| `core-013` | **authentication failed** | *The error message says application failed after three incorrect login attempts.* |
| `core-015` | **meeting continued** | *Taylor closed the door. The phone rang and then the meeting ended.* |
| `core-019` | **Office Guest** | *The Wi-Fi network name is OfficeGuest and the connection code is Alpha79.* |
| `core-022` | **application crashed** | *The application collapsed because the database connection turned out.* |
| `core-022` | **database connection timed out** | *The application collapsed because the database connection turned out.* |
