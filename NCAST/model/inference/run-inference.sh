#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --account=wiser-ewsa
#SBATCH -o /home/users/mendrika/NCAST/Output/submission-logs/inference/output/%j.out
#SBATCH -e /home/users/mendrika/NCAST/Output/submission-logs/inference/error/%j.err

set -euo pipefail

echo " Job started on $(hostname) at $(date)"

# environment
module load jaspy/3.11
source /home/users/mendrika/virtual-env/DeepLearning/bin/activate

# parameters
lead_time=$1
year=$2
month=$3
hour=$4

# put the correct path here
script=/home/users/mendrika/NCAST/NCAST/model/inference/ablation/notimefeat.py

# verify script exists
if [ ! -f "$script" ]; then
    echo "Error: Python script not found at $script"
    exit 1
fi

echo "Running base NCAST inference:"
echo " Lead time : $lead_time"
echo " Year      : $year"
echo " Month     : $month"
echo " Hour      : $hour"

# execute
python "$script" "$lead_time" "$year" "$month" "$hour"

echo " Job completed successfully at $(date)"