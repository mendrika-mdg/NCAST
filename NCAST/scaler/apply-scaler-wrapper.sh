#!/bin/bash
set -euo pipefail

JOB_SCRIPT="/home/users/mendrika/NCAST/NCAST/scaler/apply-scaler.sh"

if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Job script not found: $JOB_SCRIPT"
    exit 1
fi

PARTITIONS=("train" "val")
LEAD_TIMES=("0" "1" "3" "6")

for PARTITION in "${PARTITIONS[@]}"; do
    for LEAD_TIME in "${LEAD_TIMES[@]}"; do

        echo "Submitting job for partition=${PARTITION}, lead_time=${LEAD_TIME}"

        sbatch -J "scale_${PARTITION}_t${LEAD_TIME}" \
            "$JOB_SCRIPT" "$PARTITION" "$LEAD_TIME"

        sleep 1

    done
done

echo "All jobs submitted successfully."