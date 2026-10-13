#!/bin/bash
# Run L's recipe (plain position loss, contingent capture, no force-advance, 4-reach H1 curriculum)
# with a longer delay on the task display (targets + go cue). The visual feedback of the hand stays
# at 70 ms and proprioception at 25 ms.
#   sbatch --partition=hx /nfs/scratch/mkashefi/reach-lab/slurm/train_L.sbatch <task_delay_ms> [extra train.py flags ...]
# Extra flags come after the base recipe, so they override it (argparse keeps the last value).
# wandb is offline by default (hx nodes have no internet -- relay with wandb_live_sync.sbatch);
# submit with --export=ALL,WANDB_MODE=online on vx to log directly.
# Run output (checkpoints, config) goes straight to scratch so it survives the job; every cache
# goes to the job's local /tmp. Nothing is written to $HOME.
#SBATCH --job-name=rl-L-delay
#SBATCH --time=47:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --output=/nfs/scratch/mkashefi/reach-lab/logs/train-%j.out
DELAY=${1:?usage: train_L.sbatch <task_delay_ms> [extra train.py flags ...]}
shift
ROOT=/nfs/scratch/mkashefi/reach-lab
cd $ROOT
# train.py names the run folder by the second; two jobs that start together would share one.
sleep $(( (SLURM_JOB_ID % 10) * 30 ))
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR=$ROOT WANDB_CACHE_DIR=/tmp/wandb-cache WANDB_CONFIG_DIR=/tmp/wandb-config
export WANDB_ENTITY=amor-fati
export MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp/xdg-cache CUDA_CACHE_PATH=/tmp/nv-cache PYTHONUNBUFFERED=1
echo "[train] job $SLURM_JOB_ID  node $(hostname)  partition $SLURM_JOB_PARTITION  task delay $DELAY ms  extra: $*  wandb $WANDB_MODE  $(date -u +%F_%T)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
BASE="--effector arm26 --arch modular --n-batch 20000 --track --lr 0.001 --task horizon_sequence --n-reaches 7 --curriculum-reaches 4 --curriculum-batches 4000 --curriculum-horizon-probs 1,0,0 --no-force-advance --batch-size 256 --w-loss-jerk 1e4 --w-loss-action 1 --w-loss-action-diff 0 --w-loss-hidden 1e-3 --w-loss-hidden-diff 1e0 --w-loss-blind-hold 1 --obs-noise 0.001 --neural-noise 0.001 --action-noise 0.0001 --prob-no-go 0.1 --prob-no-go-reach 0 --capture-mode contingent --capture-radius-cm 1.0 --capture-hold-ms 300 --reach-timeout-range-ms 600 1000 --delay-range-ms 300,1500 --ckpt-every 2000 --seed 0 --print-loss-every 100"
$ROOT/env/bin/python train.py $BASE --task-delay-ms $DELAY "$@"
echo "[train] exit $?  $(date -u +%F_%T)"
