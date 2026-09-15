#!/bin/bash
# First-time setup on the GPU server (run inside the unpacked project directory):
#   bash scripts/server_bootstrap.sh
# Follows the server manual: check occupancy first, create an isolated uv env, pick the torch
# build from the driver version, verify CUDA, and fall back to the HF mirror if huggingface.co
# is unreachable. Nothing here touches the system Python or other people's GPUs.
set -e
cd "$(dirname "$0")/.."

echo "== GPU occupancy (memory + processes, not just utilisation)"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
echo
echo "== disk / cpu / ram"
df -h "$HOME" | tail -1
echo "cpus: $(nproc)   ram: $(free -g | awk '/Mem/ {print $2}') GB"

if ! command -v uv >/dev/null; then
  echo "uv not found: installing to ~/.local/bin (user scope only)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d .venv ]; then
  uv venv --python "${PY:-3.12}" .venv
fi
source .venv/bin/activate

# torch wheel by driver version (driver >= 580 -> cu130, >= 570 -> cu128, >= 560 -> cu126, else cu124)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if   [ "$DRV" -ge 580 ]; then IDX=cu130
elif [ "$DRV" -ge 570 ]; then IDX=cu128
elif [ "$DRV" -ge 560 ]; then IDX=cu126
else IDX=cu124; fi
echo "== driver $DRV -> torch index $IDX"
uv pip install torch --index-url "https://download.pytorch.org/whl/$IDX"
uv pip install -r requirements-ml.txt

python - <<'PY'
import torch, librosa, soundfile
print("torch", torch.__version__, "cuda runtime", torch.version.cuda, "available", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

echo "== HuggingFace reachability"
if curl -s -o /dev/null -m 10 -w "%{http_code}" https://huggingface.co/api/datasets/project-riz/osu-beatmaps | grep -q 200; then
  echo "huggingface.co ok"
else
  echo "huggingface.co unreachable -> using https://hf-mirror.com (export HF_ENDPOINT=https://hf-mirror.com)"
  export HF_ENDPOINT=https://hf-mirror.com
  echo 'export HF_ENDPOINT=https://hf-mirror.com' >> .venv/bin/activate
fi
echo "bootstrap done. next: bash scripts/server_prepare.sh 74 24"
