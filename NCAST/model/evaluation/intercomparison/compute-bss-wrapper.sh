#!/bin/bash
# Submit ROCSS computation jobs for all lead times and hours

JOB_SCRIPT="/home/users/mendrika/NCAST/NCAST/model/evaluation/intercomparison/compute-bss.sh"

# Lead times to process
LEAD_TIMES=("1" "3" "6")

TARGET_HOURS=($(seq -w 0 23))

for LEAD_TIME in "${LEAD_TIMES[@]}"; do
    for HOUR in "${TARGET_HOURS[@]}"; do
        echo "Submitting job for lead_time=${LEAD_TIME}, hour=${HOUR}..."
        sbatch -J "bss_t${LEAD_TIME}_h${HOUR}" "$JOB_SCRIPT" "$LEAD_TIME" "$HOUR"
        sleep 1  # small delay to avoid submission bursts
    done
done

echo "All jobs submitted successfully."
