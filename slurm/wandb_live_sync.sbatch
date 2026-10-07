#!/bin/bash
# Relay an OFFLINE wandb run to the cloud while it is still training.
# Training runs on hx (fast, but no internet) and writes its wandb data to shared scratch; this
# small job sits on a vx node (internet, same scratch) and streams that data up with wandb's
# live sync. One CPU, no GPU. It stops by itself when the run finishes; if the training job dies
# uncleanly the live sync would wait forever, so it also watches the training log and stops then.
#   sbatch /nfs/scratch/mkashefi/reach-lab/slurm/wandb_live_sync.sbatch <offline-run-dir> <training job id>
#SBATCH --job-name=rl-wandb-sync
#SBATCH --partition=vx
#SBATCH --time=47:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/nfs/scratch/mkashefi/reach-lab/logs/wandb-sync-%j.out
RUN=${1:?usage: wandb_live_sync.sbatch <offline-run-dir> <training job id>}
TRAIN_JOB=${2:-}
ROOT=/nfs/scratch/mkashefi/reach-lab
export WANDB_CACHE_DIR=/tmp/wandb-cache WANDB_CONFIG_DIR=/tmp/wandb-config XDG_CACHE_HOME=/tmp/xdg-cache
WANDB="$ROOT/env/bin/wandb"
echo "[sync] job $SLURM_JOB_ID on $(hostname): $RUN  (watching training job ${TRAIN_JOB:-none})  $(date -u +%F_%T)"
$WANDB beta sync --live --yes -e amor-fati -p arm-rnn "$RUN" &
SYNC=$!
# The training job writes a progress line to its log every few seconds, so a log that has not
# changed for 15 minutes means training is over (finished, cancelled or dead). This does not rely
# on sacct, which returned nothing on the vx nodes when it was tried (2026-10-06).
LOG=$ROOT/logs/train-$TRAIN_JOB.out
while kill -0 $SYNC 2>/dev/null; do
    sleep 60
    if [ -n "$TRAIN_JOB" ] && [ -f "$LOG" ]; then
        idle=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
        if [ "$idle" -gt 900 ]; then
            echo "[sync] training log idle for ${idle}s -> training is over; stopping the live sync  $(date -u +%F_%T)"
            kill $SYNC 2>/dev/null; break
        fi
    fi
done
wait $SYNC 2>/dev/null
echo "[sync] live sync ended; final pass  $(date -u +%F_%T)"
$WANDB beta sync --yes --no-skip-synced -e amor-fati -p arm-rnn "$RUN"
echo "[sync] done  $(date -u +%F_%T)"
