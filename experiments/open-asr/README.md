# Open ASR baseline workflow

This workflow executes the reviewed Zipformer Space code and the official Open
ASR normalizer/scorer; it does not implement a competing WER. Exact source,
dataset, model, and Icefall revisions are recorded in
`../../docs/open_asr_leaderboard_contract.md` and enforced by the runner.

## Important boundaries

- The baseline model is `CC-BY-NC-4.0`. It is useful for research and
  comparison, but it cannot be the commercial foundation of DeafBench without
  separate permission.
- The pinned published model result is 5.56 public seven-set macro WER. The
  user's 5.37 target was not the value in the pinned official result artifact.
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

The seven public sets total 74,842 rows and 161.316 hours. Their selected
Parquet configs are about 19.4 GB. The completed full LibriSpeech test-clean
run scored **1.31% official WER** over 2,620 rows, reached 97.3391 RTFx, took
350.725 seconds wall time, and used 6,790,409,216 peak VRAM bytes. The runner's
convenience WER for the same manifest was 1.67%; it is not the leaderboard
score. See `results/zipformer-librispeech-clean-full-score.json` for the
machine-readable official result.
