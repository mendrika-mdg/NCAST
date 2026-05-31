#!/bin/bash
# wrapper to submit ensemble training jobs for all lead times and seeds

JOB_SCRIPT="/home/users/mendrika/NCAST/NCAST/model/base/train.sh"

# lead times and seeds
# LEAD_TIMES=("1" "3" "6")
# SEEDS=(1 134 676 1998 2025)

LEAD_TIMES=("1" )
SEEDS=(40 50)

# loop over all combinations
for LEAD_TIME in "${LEAD_TIMES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Submitting job for lead_time=${LEAD_TIME}, seed=${SEED}..."
        sbatch -J ens_t${LEAD_TIME}_s${SEED} "$JOB_SCRIPT" "$LEAD_TIME" "$SEED"
        sleep 2 # avoid flooding the scheduler
    done
done

echo "All ensemble jobs submitted successfully."
