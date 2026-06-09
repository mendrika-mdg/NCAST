#!/bin/bash

JOB_SCRIPT="/home/users/mendrika/NCAST/NCAST/model/evaluation/ensemble/compute-fss.sh"

LEAD_TIMES=("1" "3" "6")

for LEAD_TIME in "${LEAD_TIMES[@]}"; do

    JOB_NAME="fss-ensemble-t${LEAD_TIME}"

    echo "Submitting ${JOB_NAME}"

    sbatch -J "${JOB_NAME}" \
           "${JOB_SCRIPT}" \
           "${LEAD_TIME}"

    sleep 0.5

done

echo "All ensemble FSS jobs submitted"