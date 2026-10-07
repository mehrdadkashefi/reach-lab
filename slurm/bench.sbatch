#!/bin/bash
# Speed test only: the real training command for a handful of batches, no wandb, nothing kept.
#   sbatch --partition=hx /nfs/scratch/mkashefi/reach-lab/slurm/bench.sbatch
#SBATCH --job-name=rl-bench
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --output=/nfs/scratch/mkashefi/reach-lab/logs/bench-%j.out
ROOT=/nfs/scratch/mkashefi/reach-lab
cd $ROOT
export MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/xdg-cache CUDA_CACHE_PATH=/tmp/nv-cache WANDB_MODE=disabled PYTHONUNBUFFERED=1
echo "node $(hostname)  partition $SLURM_JOB_PARTITION  $(lscpu | grep 'Model name' | sed 's/.*: *//')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$ROOT/env/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
BASE="--effector arm26 --arch modular --lr 0.001 --task horizon_sequence --n-reaches 7 --no-force-advance --batch-size 224 --w-loss-jerk 1e4 --w-loss-action 1 --w-loss-action-diff 0 --w-loss-hidden 1e-3 --w-loss-hidden-diff 1e0 --obs-noise 0.001 --neural-noise 0.001 --action-noise 0.0001 --prob-no-go 0.1 --prob-no-go-reach 0 --capture-mode contingent --capture-radius-cm 1.0 --capture-hold-ms 300 --reach-timeout-range-ms 900 1300 --delay-range-ms 300,1500 --w-loss-blind-hold 1 --pos-deadzone-cm 1.0 --pos-ramp-per-step 0.1 --pos-ramp-start-ms 100 --pos-grace-ms 50 --pos-ramp-cap 20 --pos-hold-weight 5 --w-loss-vel 0.1 --reach-len-range-cm 3.5 6 --reach-len-mix-frac 0.5 --reach-len-long-min-cm 12 --pulse-prob 0.004 --pulse-force-n 8 --pulse-dur-ms 100 --target-noise-cm 0 --snap-every 1000"
ls experiments | sort > /tmp/before.txt
echo "---- short stage (4 reaches, 810 steps)"
$ROOT/env/bin/python train.py $BASE --curriculum-reaches 4 --curriculum-batches 4000 --curriculum-horizon-probs 1,0,0 --shape-start-batch 4000 --shape-ramp-batches 2000 --n-batch 14 2>&1 | tr '\r' '\n' | grep -E "it/s|s/it|Error|Traceback" | tail -n 1
echo "---- full length (7 reaches, 1200 steps, all terms on)"
$ROOT/env/bin/python train.py $BASE --n-batch 10 2>&1 | tr '\r' '\n' | grep -E "it/s|s/it|Error|Traceback" | tail -n 1
for d in $(ls experiments | sort | comm -13 /tmp/before.txt -); do rm -rf "experiments/$d"; echo "removed test folder experiments/$d"; done
