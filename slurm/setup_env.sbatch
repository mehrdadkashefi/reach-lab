#!/bin/bash
# One-off setup on the CBS SLURM cluster: build the Python environment in scratch and time the
# training loop on an A100. Runs on a vx node (the partition with internet), never on the login
# node. Nothing is written to $HOME: every cache goes to the job's local /tmp, which the cluster
# wipes when the job ends.
#   submit:  sbatch /nfs/scratch/mkashefi/reach-lab/slurm/setup_env.sbatch
#SBATCH --job-name=rl-setup
#SBATCH --partition=vx
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --output=/nfs/scratch/mkashefi/reach-lab/logs/setup-%j.out

ROOT=/nfs/scratch/mkashefi/reach-lab
ENV=$ROOT/env
source /srv/software/user_init.sh            # the cluster's module setup
module load uv/0.12.23                        # the cluster's own installer tool
set -eo pipefail
export UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy XDG_CACHE_HOME=/tmp/xdg-cache
echo "node $(hostname)   $(date -u +%F_%T)   uv $(uv --version)"

# ---- environment: the cluster's system Python 3.12 + the lab server's package versions
if [ ! -x $ENV/bin/python ]; then
    uv venv --python /usr/bin/python3 $ENV
fi
# PyTorch from its own CUDA 12.6 wheel index ONLY. The nodes run NVIDIA driver 570, which is too
# old for the CUDA 13 build that plain PyPI serves for torch 2.13 (and uv gives --extra-index-url
# priority over --index-url, so naming PyPI as an extra index silently picks that build).
uv pip install --python $ENV/bin/python \
    --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126"
uv pip install --python $ENV/bin/python \
    numpy==2.4.6 matplotlib==3.11.1 seaborn==0.13.2 pandas==3.0.5 tqdm==4.70.0 wandb==0.28.1
echo "---- installed"
uv pip list --python $ENV/bin/python
du -sh $ENV
$ENV/bin/python - <<'PY'
import torch, numpy, matplotlib, wandb, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda available", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", "| numpy", numpy.__version__, "| wandb", wandb.__version__)
assert torch.cuda.is_available(), "GPU not usable from this environment -- stopping before the speed test"
PY

# ---- speed test: the real training command, a handful of batches, no wandb
cd $ROOT
export MPLCONFIGDIR=/tmp/mpl CUDA_CACHE_PATH=/tmp/nv-cache WANDB_MODE=disabled PYTHONUNBUFFERED=1
BASE="--effector arm26 --arch modular --lr 0.001 --task horizon_sequence --n-reaches 7 --no-force-advance --batch-size 224 --w-loss-jerk 1e4 --w-loss-action 1 --w-loss-action-diff 0 --w-loss-hidden 1e-3 --w-loss-hidden-diff 1e0 --obs-noise 0.001 --neural-noise 0.001 --action-noise 0.0001 --prob-no-go 0.1 --prob-no-go-reach 0 --capture-mode contingent --capture-radius-cm 1.0 --capture-hold-ms 300 --reach-timeout-range-ms 900 1300 --delay-range-ms 300,1500 --w-loss-blind-hold 1 --pos-deadzone-cm 1.0 --pos-ramp-per-step 0.1 --pos-ramp-start-ms 100 --pos-grace-ms 50 --pos-ramp-cap 20 --pos-hold-weight 5 --w-loss-vel 0.1 --reach-len-range-cm 3.5 6 --reach-len-mix-frac 0.5 --reach-len-long-min-cm 12 --pulse-prob 0.004 --pulse-force-n 8 --pulse-dur-ms 100 --target-noise-cm 0 --snap-every 1000"
ls experiments 2>/dev/null | sort > /tmp/before.txt || true
echo "---- speed, short stage (4 reaches, 810 steps)"
$ENV/bin/python train.py $BASE --curriculum-reaches 4 --curriculum-batches 4000 --curriculum-horizon-probs 1,0,0 --shape-start-batch 4000 --shape-ramp-batches 2000 --n-batch 14 2>&1 | tr '\r' '\n' | grep -E "it/s|s/it|Error|Traceback" | tail -n 2
echo "---- speed, full length (7 reaches, 1200 steps, all terms on)"
$ENV/bin/python train.py $BASE --n-batch 10 2>&1 | tr '\r' '\n' | grep -E "it/s|s/it|Error|Traceback" | tail -n 2
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
# remove the two throw-away run folders the speed test created
for d in $(ls experiments | sort | comm -13 /tmp/before.txt -); do rm -rf "experiments/$d"; echo "removed test folder experiments/$d"; done
echo "done $(date -u +%F_%T)"
