#!/bin/bash

lead_time=1

configs=(
  "1e-4 0.2 25.0 0.3"
  "5e-5 0.2 25.0 0.3"
  "2e-4 0.2 25.0 0.3"

  "1e-4 0.1 25.0 0.3"
  "1e-4 0.3 25.0 0.3"

  "1e-4 0.2 10.0 0.3"
  "1e-4 0.2 50.0 0.3"

  "1e-4 0.2 25.0 0.1"
  "1e-4 0.2 25.0 0.5"
)

for cfg in "${configs[@]}"
do
  read lr dropout_p pos_weight alpha <<< "$cfg"

  echo "Submitting:"
  echo "lead_time=${lead_time}"
  echo "lr=${lr}"
  echo "dropout_p=${dropout_p}"
  echo "pos_weight=${pos_weight}"
  echo "alpha=${alpha}"

  sbatch \
  --job-name=t${lead_time}_lr${lr}_do${dropout_p}_pw${pos_weight}_a${alpha} \
  /home/users/mendrika/NCAST/NCAST/model/hyperparam-tuning/ncast-opt.sh \
  "$lead_time" \
  "$lr" \
  "$dropout_p" \
  "$pos_weight" \
  "$alpha"

  sleep 1
done