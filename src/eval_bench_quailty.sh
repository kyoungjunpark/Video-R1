#!/bin/bash
# run_models.sh
#     # "src/r1-v/log/Qwen2.5-VL-7B-Video-7B-cot-sft/",
model_paths=(
    "src/r1-v/log/Qwen2.5-VL-7B-Quality-GRPO-match_format-temgen/checkpoint-2000"
)

file_names=(
    "FileName"
)
output_path=("src/r1-v/Video-Ours-data/grpo_output_quality_gentemp.json")
dataset_path=("src/r1-v/Video-Ours-data/real_gen_r1_sft_cot_quality_test.json")

export DECORD_EOF_RETRY_MAX=20480


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    CUDA_VISIBLE_DEVICES=0 python ./src/eval_bench_quality.py --model_path "$model" --file_name "$file_name"
done
