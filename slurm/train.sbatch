#!/bin/bash
# One training run of the phased-in recipe on the CBS SLURM cluster.
#   sbatch --partition=hx /nfs/scratch/mkashefi/reach-lab/slurm/train.sbatch <target_noise_cm> [wandb_mode]
# wandb_mode: "offline" (default; hx nodes have no internet -- sync afterwards) or "online" (vx).
# Measured 2026-10-06 at batch 224: hx (L40S, Xeon 6448Y) 3.9 s/batch short stage, 6.2 full
# length -> ~32 h for 20,000 batches; vx (A100, Xeon 6142) 6.0 / 9.3 -> ~48 h, i.e. the job limit.
# Run output (checkpoints, config) goes straight to scratch so it survives the job; every cache
# goes to the job's local /tmp. Nothing is written to $HOME.
#SBATCH --job-name=rl-train
#SBATCH --time=47:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --output=/nfs/scratch/mkashefi/reach-lab/logs/train-%j.out
NOISE=${1:?usage: train.sbatch <target_noise_cm> [offline|online]}
ROOT=/nfs/scratch/mkashefi/reach-lab
cd $ROOT
export WANDB_MODE=${2:-offline} WANDB_DIR=$ROOT WANDB_CACHE_DIR=/tmp/wandb-cache WANDB_CONFIG_DIR=/tmp/wandb-config
export MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/xdg-cache CUDA_CACHE_PATH=/tmp/nv-cache PYTHONUNBUFFERED=1
echo "[train] job $SLURM_JOB_ID  node $(hostname)  partition $SLURM_JOB_PARTITION  noise $NOISE cm  wandb $WANDB_MODE  $(date -u +%F_%T)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
BASE="--effector arm26 --arch modular --n-batch 20000 --track --lr 0.001 --task horizon_sequence --n-reaches 7 --curriculum-reaches 4 --curriculum-batches 4000 --curriculum-horizon-probs 1,0,0 --no-force-advance --batch-size 224 --w-loss-jerk 1e4 --w-loss-action 1 --w-loss-action-diff 0 --w-loss-hidden 1e-3 --w-loss-hidden-diff 1e0 --obs-noise 0.001 --neural-noise 0.001 --action-noise 0.0001 --prob-no-go 0.1 --prob-no-go-reach 0 --capture-mode contingent --capture-radius-cm 1.0 --capture-hold-ms 300 --reach-timeout-range-ms 900 1300 --delay-range-ms 300,1500 --w-loss-blind-hold 1 --ckpt-every 1000 --pos-deadzone-cm 1.0 --pos-ramp-per-step 0.1 --pos-ramp-start-ms 100 --pos-grace-ms 50 --pos-ramp-cap 20 --pos-hold-weight 5 --w-loss-vel 0.1 --reach-len-range-cm 3.5 6 --reach-len-mix-frac 0.5 --reach-len-long-min-cm 12 --pulse-prob 0.004 --pulse-force-n 8 --pulse-dur-ms 100 --shape-start-batch 4000 --shape-ramp-batches 2000 --print-loss-every 100"
$ROOT/env/bin/python train.py $BASE --target-noise-cm $NOISE
echo "[train] exit $?  $(date -u +%F_%T)"
