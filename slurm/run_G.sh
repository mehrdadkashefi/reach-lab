#!/bin/bash
# Lab-server launcher: run L's recipe with a 200 ms task-display delay and SHORT training reaches
# (every reach 2.5-3.5 cm, the monkey's 3 cm spacing), with or without a hand-speed cost.
#   bash run_G.sh <w_loss_vel> <log file> [extra train.py flags ...]   (run inside tmux; pin the GPU with CUDA_VISIBLE_DEVICES)
# The speed cost is off during the 4-reach stage and ramps in over batches 4000-6000.
# Batch 224 and a 800-1200 ms per-reach budget (episode 1130 steps) fit the 11 GB cards.
W=${1:?usage: run_G.sh <w_loss_vel> <log file> [extra train.py flags ...]}; LOG=${2:?log file}; shift 2
cd ~/projects/reach-lab
source /software/anaconda3/etc/profile.d/conda.sh && conda activate reach-lab
export WANDB_ENTITY=amor-fati PYTHONUNBUFFERED=1
python train.py --effector arm26 --arch modular --n-batch 20000 --track --lr 0.001 \
  --task horizon_sequence --n-reaches 7 --curriculum-reaches 4 --curriculum-batches 4000 \
  --curriculum-horizon-probs 1,0,0 --no-force-advance --batch-size 224 \
  --w-loss-jerk 1e4 --w-loss-action 1 --w-loss-action-diff 0 --w-loss-hidden 1e-3 \
  --w-loss-hidden-diff 1e0 --w-loss-blind-hold 1 --obs-noise 0.001 --neural-noise 0.001 \
  --action-noise 0.0001 --prob-no-go 0.1 --prob-no-go-reach 0 --capture-mode contingent \
  --capture-radius-cm 1.0 --capture-hold-ms 300 --reach-timeout-range-ms 800 1200 \
  --delay-range-ms 300,1500 --ckpt-every 2000 --seed 0 --print-loss-every 100 \
  --task-delay-ms 200 --reach-len-range-cm 2.5 3.5 \
  --w-loss-vel $W --shape-start-batch 4000 --shape-ramp-batches 2000 "$@" > $LOG 2>&1
echo "exit $? $(date -u +%F_%T)" >> $LOG
