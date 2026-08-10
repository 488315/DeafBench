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
wsl -d archlinux -- bash /mnt/c/Users/kjones/Documents/DeafBench/experiments/open-asr/run_zipformer_wsl.sh librispeech test.clean 2
```

The command downloads or reuses the pinned checkpoint, materializes the pinned
dataset config, decodes two rows, and writes an official JSONL manifest below
`~/.cache/deafbench/open-asr-runs/results/`.

Score the copied manifests from Windows with the pinned official checkout:

```powershell
python -m deafbench leaderboard score experiments\open-asr\results `
  --official-repo C:\path\to\open-asr-official-9585-clean `
  --model-id soundsgoodai/Zipformer-cr-ctc-transducer-XL-290M
```

Use `full` as the third argument only at a defined public-test milestone:

```powershell
wsl -d archlinux -- bash /mnt/c/Users/kjones/Documents/DeafBench/experiments/open-asr/run_zipformer_wsl.sh librispeech test.clean full
```

The seven public sets contain 74,842 raw rows and 161.316 hours. The official
loader may remove references that normalize to empty or its ignore sentinel;
Earnings22 therefore emits 2,737 result rows from 2,741 raw rows. Their selected
Parquet configs are about 19.4 GB. The completed full LibriSpeech test-clean
run scored **1.31% official WER** over 2,620 rows, reached 97.3391 RTFx, took
350.725 seconds wall time, and used 6,790,409,216 peak VRAM bytes. The runner's
convenience WER for the same manifest was 1.67%; it is not the leaderboard
score. The completed VoxPopuli set scored **4.31% official WER** over 628 rows
at 18.8164 RTFx. Earnings22 scored **7.68% official WER** over 2,737 evaluable
rows at 86.5395 RTFx and 4,220,642,304 peak VRAM bytes. The three-set mean is
4.43%. LibriSpeech test-other scored **3.01% official WER** over 2,939 rows,
bringing the four-set mean to 4.08%; neither partial mean is the seven-set
composite. See `results/zipformer-public-4set-score.json` for the current
machine-readable result.
