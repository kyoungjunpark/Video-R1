#!/bin/bash
# run_models.sh
#     # "src/r1-v/log/Qwen2.5-VL-7B-Video-7B-cot-sft/",
model_paths=(
    # "src/r1-v/log/Qwen2.5-VL-7B-GRPO-gentemp/checkpoint-6000"
    # "/blob/kyoungjun/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-GRPO-gentemp/checkpoint-6000"
    # "/blob/kyoungjun/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-Video-7B-cot-sft/"
    # "/blob/kyoungjun/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-GRPO-gentemp2/checkpoint-1500"
    # "Qwen/Qwen2.5-VL-7B-Instruct"
    # "/blob/kyoungjun/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-GRPO-genvideo/checkpoint-600"
    # "src/r1-v/log/Qwen2.5-VL-7B-GRPO-0527/checkpoint-10000"
    # "/blob/kyoungjun/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-GRPO-cogvideo-0605/checkpoint-10000"
    "/blob/kyoungjun/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-Video-7B-newcot-sft-0612/checkpoint-10000"
    # "/blob/kyoungjun/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-GRPO-genvidbench/checkpoint-800"
    # "/blob/kyoungjun/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-GRPO-gentemp-0521-still/checkpoint-2000"
   #  "/blob/kyoungjun/Video-R1/src/r1-v/log/Qwen2.5-VL-7B-GRPO-ourdata-genvideo-0522/checkpoint-2250"
)

file_names=(
    "FileName"
)
# output_path=("src/r1-v/Video-Ours-data/our_zeroshot_genvieo.json")
# output_path=("src/r1-v/Video-Ours-data/our_GRPO_gentemp2.json")
output_path=("src/r1-v/Video-Ours-data/our_cot.json")
dataset_path=("/blob/kyoungjun/Video-R1/src/r1-v/Video-Ours-data/real_gen_r1_grpo_test.json")

export DECORD_EOF_RETRY_MAX=20480


for i in "${!model_paths[@]}"; do
    model="${model_paths[$i]}"
    file_name="${file_names[$i]}"
    CUDA_VISIBLE_DEVICES=1 python ./src/eval_bench.py --model_path "$model" --file_name "$file_name" --output_path "$output_path" --dataset_path "$dataset_path"
done
