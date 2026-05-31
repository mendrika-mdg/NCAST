#!/bin/bash
set -e

for lead_time in 1 3 6
do
    echo "Submitting no-attention ablation for lead_time=${lead_time}"

    sbatch \
        --job-name=ncast-noatt-t${lead_time} \
        /home/users/mendrika/NCAST/NCAST/model/ablation/train-attention.sh \
        ${lead_time}

    sleep 1
done