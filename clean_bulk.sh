#!/bin/bash
COHORT=$1

gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader)
unique_gpu_names=$(echo "$gpu_info" | awk -F':' '!seen[$NF]++ {gsub(/^[ \t]+|[ \t]+$/, "", $NF); print $NF}')
gpu_name=$(echo "$unique_gpu_names" | tr -d '[:space:]')

python -W ignore correct_ocr_mistral.py \
--cohort $COHORT \
--gpu $gpu_name
