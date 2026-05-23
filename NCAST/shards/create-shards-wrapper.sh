#!/bin/bash
# launcher for data shard creation

JOB_SCRIPT="/home/users/mendrika/NCAST/NCAST/shards/create-shards.sh"

# lead times and partitions
LEAD_TIMES=("0" "1" "3" "6")
PARTITIONS=("train" "val")

for LEAD_TIME in "${LEAD_TIMES[@]}"; do
    for PARTITION in "${PARTITIONS[@]}"; do
        echo "Submitting job for partition=${PARTITION}, lead_time=${LEAD_TIME}..."
        sbatch -J "${PARTITION}${LEAD_TIME}" "$JOB_SCRIPT" "$PARTITION" "$LEAD_TIME"
        sleep 1
    done
done

echo "All jobs submitted successfully."

