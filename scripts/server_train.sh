#!/bin/bash
# Train both paradigms in parallel as two DDP jobs (torchrun, one process per card).
#
#   export WANDB_API_KEY=...   (or `wandb login` once)
#   bash scripts/server_train.sh <gpus_for_ar> <gpus_for_masked> [wandb_project]
#   e.g. bash scripts/server_train.sh 4,5 6,7 autoosu-rhythm
#   v1 (hear-ahead audio context): SUFFIX=v1 LAYERS=6 EXTRA="--audio-ctx-layers 4" bash scripts/server_train.sh 4,5 6,7
#
# Cards are selected by UUID (manual section 4), NCCL_P2P_DISABLE=1 is the server's known-good
# setting, and numactl pins each job to the NUMA node of its cards when the whole group sits on one
# node (0-3 -> node 0, 4-7 -> node 1). Runs live in tmux sessions "ar" and "masked".
set -e
export PATH="$HOME/.local/bin:$PATH"   # uv lives here when installed by the bootstrap
GPUS_AR=${1:?comma separated gpu indices for the ar run, e.g. 4,5}
GPUS_MASKED=${2:?comma separated gpu indices for the masked run, e.g. 6,7}
PROJECT=${3:-autoosu-rhythm}
SUFFIX=${SUFFIX:-v0}          # run directory suffix: runs/ar_<SUFFIX>, runs/masked_<SUFFIX>
LAYERS=${LAYERS:-8}
EXTRA=${EXTRA:-}              # extra train.py args, e.g. EXTRA="--audio-ctx-layers 4"
cd "$(dirname "$0")/.."
ROOT=$(pwd)

for g in ${GPUS_AR//,/ } ${GPUS_MASKED//,/ }; do
  used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$used" -gt 2000 ]; then
    echo "GPU $g already has ${used} MiB in use - pick another card"; exit 1
  fi
done

COMMON="--prep data/prep --steps 60000 --batch 64 --seq 1024 --d-model 512 --layers $LAYERS --heads 8 --workers 8 --eval-every 2000 --compile --wandb $PROJECT $EXTRA"

numa_prefix() {   # prints a numactl prefix if all given gpus are on one NUMA node and numactl exists
  local gpus=$1 node=""
  command -v numactl >/dev/null || return 0
  for g in ${gpus//,/ }; do
    local n; n=$(nvidia-smi topo -m 2>/dev/null | awk -v g="GPU$g" '$1==g {print $(NF-1)}')
    [[ "$n" =~ ^[0-9]+$ ]] || return 0
    if [ -z "$node" ]; then node=$n; elif [ "$node" != "$n" ]; then return 0; fi
  done
  echo "numactl --cpunodebind=$node --membind=$node"
}

launch() {   # name gpus mode
  local name=$1 gpus=$2 mode=$3
  local uuids; uuids=$(nvidia-smi -i "$gpus" --query-gpu=uuid --format=csv,noheader | paste -sd, -)
  local n; n=$(echo "$uuids" | tr ',' '\n' | wc -l)
  local numa; numa=$(numa_prefix "$gpus")
  mkdir -p "runs/${name}_${SUFFIX}"
  tmux new -d -s "$name" "cd $ROOT && source .venv/bin/activate && export CUDA_VISIBLE_DEVICES=$uuids NCCL_P2P_DISABLE=1 && \
    $numa torchrun --standalone --nnodes=1 --nproc-per-node=$n -m autoosu.ml.train --mode $mode \
    --out runs/${name}_${SUFFIX} --run-name ${name}-${SUFFIX}-${n}gpu $COMMON 2>&1 | tee runs/${name}_${SUFFIX}/train.log"
  echo "started tmux session '$name': mode $mode on GPUs $gpus ($n procs) ${numa:+with $numa}"
}

launch ar "$GPUS_AR" ar
launch masked "$GPUS_MASKED" masked
echo "attach with: tmux attach -t ar   |   tmux attach -t masked"
