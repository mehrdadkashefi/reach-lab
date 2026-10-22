#!/bin/bash
# Lab-server launcher: continue a trained B-recipe network (dwell wall 3 + dwell-speed cost, 200 ms
# task-display delay, short reaches) for more batches.
#   bash run_continue.sh <checkpoint .pt> <data seed> <log file> [extra train.py flags ...]
#   (run inside tmux; pin the GPU with CUDA_VISIBLE_DEVICES)
# --init-from loads the weights and the connectivity masks (they are buffers), so the data seed only
# sets the new training data. The 4-reach curriculum and the shape ramp are switched off: the full
# 7-reach task and every cost are on from the first batch. Adam starts fresh.
CKPT=${1:?usage: run_continue.sh <checkpoint .pt> <data seed> <log file> [extra train.py flags ...]}
SEED=${2:?data seed}; LOG=${3:?log file}; shift 3
cd ~/projects/reach-lab
PY=python
[ -f /software/anaconda3/etc/profile.d/conda.sh ] && source /software/anaconda3/etc/profile.d/conda.sh && conda activate reach-lab \
  || PY=~/.conda/envs/reach-lab/bin/python
export WANDB_ENTITY=amor-fati PYTHONUNBUFFERED=1
$PY train.py --effector arm26 --arch modular --n-batch 10000 --track --lr 0.001 \
  --task horizon_sequence --n-reaches 7 --curriculum-batches 0 --no-force-advance --batch-size 224 \
  --w-loss-jerk 1e4 --w-loss-action 1 --w-loss-action-diff 0 --w-loss-hidden 1e-3 \
  --w-loss-hidden-diff 1e0 --w-loss-blind-hold 1 --obs-noise 0.001 --neural-noise 0.001 \
  --action-noise 0.0001 --prob-no-go 0.1 --prob-no-go-reach 0 --capture-mode contingent \
  --capture-radius-cm 1.0 --capture-hold-ms 300 --reach-timeout-range-ms 800 1200 \
  --delay-range-ms 300,1500 --ckpt-every 2000 --seed $SEED --print-loss-every 100 \
  --task-delay-ms 200 --reach-len-range-cm 2.5 3.5 \
  --w-loss-wall 3 --w-loss-dwell-speed 2 --dwell-speed-cm-s 2.5 --dwell-speed-grace-ms 150 \
  --shape-start-batch 0 --shape-ramp-batches 0 --init-from $CKPT "$@" > $LOG 2>&1
echo "exit $? $(date -u +%F_%T)" >> $LOG
