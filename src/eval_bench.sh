#!/bin/bash
# run_models.sh

model_paths=(
    "src/r1-v/log/Qwen2.5-VL-7B-Video-7B-cot-sft/"
)

file_names=(
    "FileName"
)

export DECORD_EOF_RETRY_MAX=20480


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    CUDA_VISIBLE_DEVICES=0 python ./src/eval_bench.py --model_path "$model" --file_name "$file_name"
done
