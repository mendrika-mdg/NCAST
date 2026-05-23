#!/bin/bash
# launcher for LR optimisation jobs

JOB_SCRIPT="/home/users/mendrika/NCAST/1T-Dakar/BASH/optimise.sh"

LEAD_TIMES=("1" "3" "6")

for LEAD_TIME in "${LEAD_TIMES[@]}"; do

    echo "Submitting optimisation for lead_time=${LEAD_TIME}h..."

    sbatch -J "Opt_t${LEAD_TIME}" \
        "$JOB_SCRIPT" "$LEAD_TIME"

    sleep 2

done

echo "All LR optimisation jobs submitted successfully."