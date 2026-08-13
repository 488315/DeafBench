# Open ASR baseline workflow

This workflow executes the reviewed Zipformer Space code and the official Open
ASR normalizer/scorer; it does not implement a competing WER. Exact source,
dataset, model, and Icefall revisions are recorded in
`../../docs/open_asr_leaderboard_contract.md` and enforced by the runner.

## Important boundaries

- The baseline model is `CC-BY-NC-4.0`. It is useful for research and
  comparison, but it cannot be the commercial foundation of DeafBench without
  separate permission.
- The pinned published Zipformer result is 5.56 public seven-set macro WER.
  The pinned leaderboard CSV's best public average is 5.42; 5.37 is Zoom
  Scribe's VoxPopuli column, not its average. Private success still requires
  an authorized official submission.
- A public score does not prove a leaderboard win. Private scripted and
  conversational evaluation is available only through an authorized official
  submission.
- Do not train on these test manifests or tune repeatedly against their labels.
- `evaluation-policy.json` is the machine-tested data-use boundary. It keeps
  training, development, and official evaluation disjoint and blocks a final
  claim while model-training contamination remains indeterminate.
- `evidence-manifest.json` freezes the seven public manifests, official score,
  error analysis, source revisions, decoding contract, and local RTX 4070
  label. Its integrity test must pass before these results are reported.

## Environment

Use WSL2 with Python 3.12 and the RTX 4070 visible to CUDA. From the repository
root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e . \
  -r experiments/open-asr/requirements.lock.txt
```

Create clean source checkouts at the exact revisions listed in
`submission.yaml`. The runner rejects revision drift and any modified or
untracked source. The expected locations are under
`${XDG_CACHE_HOME:-$HOME/.cache}/deafbench/`:

- `open-asr-official-9585-clean`
- `open-asr-zipformer-64c6-clean`
- `icefall`

## Small end-to-end test

From Windows PowerShell:

```powershell
$repoWsl = wsl wslpath -a $PWD
wsl -d archlinux -- bash "$repoWsl/experiments/open-asr/run_zipformer_wsl.sh" librispeech test.clean 2
```

The command downloads or reuses the pinned checkpoint, materializes the pinned
dataset config, decodes two rows, and writes an official JSONL manifest below
`~/.cache/deafbench/open-asr-runs/results/`.

The tracked two-row batch-1 diagnostic and the first two rows of the independent
batch-32 full run have identical audio identities, references, and predictions.
`tests/test_open_asr_evidence.py` verifies those stable fields while excluding
run-dependent timing from the reproducibility contract.

Score the copied manifests from Windows with the pinned official checkout:

```powershell
python -m deafbench leaderboard score experiments\open-asr\results `
  --official-repo C:\path\to\open-asr-official-9585-clean `
  --model-id soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M
```

Rank high-error utterances with the same pinned normalizer and compound-aware
alignment used by the scorer:

```powershell
python -m deafbench leaderboard analyze experiments\open-asr\results\full-public `
  --official-repo C:\path\to\open-asr-official-9585-clean `
  --model-id soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M `
  --limit 20 --output experiments\open-asr\results\zipformer-public-errors.json
```

The analyzer reports normalized reference and prediction text, per-row error
counts, and each dataset's aggregate counts. It ranks by error mass for
diagnosis; its output is not an additional leaderboard metric.

Use `full` as the third argument only at a defined public-test milestone:

```powershell
$repoWsl = wsl wslpath -a $PWD
wsl -d archlinux -- bash "$repoWsl/experiments/open-asr/run_zipformer_wsl.sh" librispeech test.clean full
```

The seven public sets contain 74,842 raw rows and 161.316 hours. The official
loader may remove references that normalize to empty or its ignore sentinel;
AMI-Cleaned emits 7,715 result rows from 7,805 raw rows, Earnings22 emits
2,737 from 2,741, and GigaSpeech-Cleaned emits 18,757 from 18,768. Their
selected Parquet configs are about 19.4 GB. The completed full LibriSpeech
test-clean
run scored **1.31% official WER** over 2,620 rows, reached 97.3391 RTFx, took
350.725 seconds wall time, and used 6,790,409,216 peak VRAM bytes. The runner's
convenience WER for the same manifest was 1.67%; it is not the leaderboard
score. The completed VoxPopuli set scored **4.31% official WER** over 628 rows
at 18.8164 RTFx. Earnings22 scored **7.68% official WER** over 2,737 evaluable
rows at 86.5395 RTFx and 4,220,642,304 peak VRAM bytes. The three-set mean is
4.43%. LibriSpeech test-other scored **3.01% official WER** over 2,939 rows,
bringing the four-set mean to 4.08%; neither partial mean is the seven-set
composite. AMI-Cleaned scored **10.30% official WER** over 7,715 evaluable
rows at 89.4465 RTFx and 2,829,196,800 peak VRAM bytes, bringing the five-set
mean to 5.32%. GigaSpeech-Cleaned scored **8.33% official WER** over 18,757
evaluable rows at 99.2171 RTFx and 2,567,505,408 peak VRAM bytes, bringing the
six-set mean to 5.82%. SPGISpeech scored **1.64% official WER** over all
39,341 rows at 100.6914 RTFx, 5,451.722 seconds wall time, and 3,062,188,544
peak VRAM bytes. The resulting complete seven-set composite is **5.23%**, as
recorded in `results/zipformer-public-7set-score.json`. This reproduced public
score is 0.19 points below the pinned leaderboard snapshot's 5.42 best public
average and 0.33 below Zipformer's published 5.56, but it does not establish a
private leaderboard win and the checkpoint's noncommercial license prevents
using it as DeafBench's commercial foundation without separate permission.

## Contamination status and next evaluation milestone

No sample-level overlap has been confirmed for the reproduced Zipformer
baseline. Its published materials do not provide a complete sample-level
training inventory, however, so contamination status is **indeterminate**, not
clear. This result is therefore baseline compatibility evidence and is not
eligible to support a final model claim.

Before evaluating a future DeafBench candidate, record immutable revisions for
every training and development source and compare stable sample or audio-content
identities against every official evaluation manifest. Select model and decoding
changes using disjoint development data. The next official public-test run must
be a predeclared candidate milestone, not another tuning iteration.
