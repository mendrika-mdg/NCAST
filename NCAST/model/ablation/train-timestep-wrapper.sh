#!/bin/bash
set -e

ablations=(
    "t0"
    "t0_tminus1h"
    "t0_tminus1h_tminus2h"
    "all"
)

for lead_time in 1 3 6
do
    for ablation_name in "${ablations[@]}"
    do
        echo "Submitting temporal ablation:"
        echo "lead_time=${lead_time}"
        echo "ablation=${ablation_name}"

        sbatch \
            --job-name=temp-t${lead_time}-${ablation_name} \
            /home/users/mendrika/NCAST/NCAST/model/ablation/train-timestep.sh \
            ${lead_time} \
            ${ablation_name}

        sleep 1
    done
done