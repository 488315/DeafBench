#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-librispeech}"
split="${2:-test.clean}"
sample_limit="${3:-2}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
venv_dir="${repo_root}/.venv"
cache_root="${XDG_CACHE_HOME:-${HOME}/.cache}/deafbench"
site_packages="${venv_dir}/lib/python3.12/site-packages"

cuda_libraries=(
  cublas cuda_cupti cuda_nvrtc cuda_runtime cudnn cufft
  curand cusolver cusparse nccl nvjitlink nvtx
)
library_path=""
for package in "${cuda_libraries[@]}"; do
  directory="${site_packages}/nvidia/${package}/lib"
  library_path="${library_path:+${library_path}:}${directory}"
done

limit_args=()
if [[ "${sample_limit}" != "full" ]]; then
  limit_args=(--max-eval-samples "${sample_limit}")
fi

export LD_LIBRARY_PATH="${library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONDONTWRITEBYTECODE=1

exec "${venv_dir}/bin/python" -m deafbench.leaderboard.zipformer_runner \
  --runner-repo "${cache_root}/open-asr-zipformer-64c6-clean" \
  --official-repo "${cache_root}/open-asr-official-9585-clean" \
  --icefall-repo "${cache_root}/icefall" \
  --output-dir "${DEAFBENCH_RUN_DIR:-${cache_root}/open-asr-runs}" \
  --dataset "${dataset}" \
  --split "${split}" \
  --batch-size "${DEAFBENCH_BATCH_SIZE:-32}" \
  --warmup-steps 1 \
  "${limit_args[@]}"
