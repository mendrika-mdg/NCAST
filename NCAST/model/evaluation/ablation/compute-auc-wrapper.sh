#!/bin/bash

JOB_SCRIPT="/home/users/mendrika/NCAST/NCAST/model/evaluation/compute-auc.sh"

MODELS=(
    "base"
    "noattention"
    "notimefeat"
    "t0"
    "t0_tminus1h"
    "t0_tminus1h_tminus2h"
)

LEAD_TIMES=("1" "3" "6")

for MODEL_NAME in "${MODELS[@]}"; do
    for LEAD_TIME in "${LEAD_TIMES[@]}"; do

        JOB_NAME="auc-${MODEL_NAME}-t${LEAD_TIME}"

        echo "Submitting ${JOB_NAME}"

        sbatch -J "${JOB_NAME}" \
               "${JOB_SCRIPT}" \
               "${MODEL_NAME}" \
               "${LEAD_TIME}"

        sleep 0.5

    done
done

echo "All AUC jobs submitted"