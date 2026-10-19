#!/bin/bash
# Relay for a training job that has not started yet: submit it with a dependency on that job, it
# waits for the job's log to name its offline wandb run, then runs wandb_live_sync.sbatch on it.
#   sbatch --dependency=after:<training job id> /nfs/scratch/mkashefi/reach-lab/slurm/wandb_relay_after.sbatch <training job id>
#SBATCH --job-name=rl-wandb-sync
#SBATCH --partition=vx
#SBATCH --time=47:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/nfs/scratch/mkashefi/reach-lab/logs/wandb-sync-%j.out
TRAIN_JOB=${1:?usage: wandb_relay_after.sbatch <training job id>}
ROOT=/nfs/scratch/mkashefi/reach-lab
LOG=$ROOT/logs/train-$TRAIN_JOB.out
# the training job staggers its start by up to 270 s before wandb names the run
for i in $(seq 120); do
    RUN=$(grep -o "$ROOT/wandb/offline-run-[^ ]*" "$LOG" 2>/dev/null | head -1)
    [ -n "$RUN" ] && break
    sleep 15
done
if [ -z "$RUN" ]; then
    echo "[sync] no offline run named in $LOG after 30 min; giving up  $(date -u +%F_%T)"
    exit 1
fi
exec bash $ROOT/slurm/wandb_live_sync.sbatch "$RUN" "$TRAIN_JOB"
