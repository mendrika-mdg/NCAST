#!/bin/bash

JOB_SCRIPT="/home/users/mendrika/NCAST/NCAST/model/inference/run-inference-vartimestep.sh"

YEARS=("2020" "2021" "2022" "2023" "2024")
MONTHS=("06" "07" "08" "09")
HOURS=($(seq -w 0 23))
LEAD_TIMES=("1" "3" "6")

ABLATIONS=(
    "t0"
    "t0_tminus1h"
    "t0_tminus1h_tminus2h"
)

for ABLATION in "${ABLATIONS[@]}"; do
    for YEAR in "${YEARS[@]}"; do
        for MONTH in "${MONTHS[@]}"; do
            for HOUR in "${HOURS[@]}"; do
                for LEAD_TIME in "${LEAD_TIMES[@]}"; do

                    JOB_NAME="vartime_${ABLATION}_t${LEAD_TIME}_${YEAR}${MONTH}_${HOUR}"

                    echo "Submitting job: ${JOB_NAME}"

                    sbatch -J "${JOB_NAME}" \
                        "$JOB_SCRIPT" \
                        "$LEAD_TIME" \
                        "$YEAR" \
                        "$MONTH" \
                        "$HOUR" \
                        "$ABLATION"

                    sleep 0.5

                done
            done
        done
    done
done

echo "All variable-timestep inference jobs submitted successfully."