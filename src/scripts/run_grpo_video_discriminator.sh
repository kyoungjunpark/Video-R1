cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export WANDB_PROJECT=Ours-GRPOTA-true-newCoT30SFT2f-06152
# For resume training:  --resume_from_checkpoint Model_Path \
# Set temporal to choose between T-GRPO and GRPO, and len_control to enable or disable the length control reward.

# Qwen/Qwen2.5-VL-7B-Instruct
# --temporal false \   # For GRPO not T-GRPO
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/grpo.py \
    --output_dir "./log/Qwen2.5-VL-7B-GRPOTA-newCoT70-2f-0616" \
    --model_name_or_path '/blob/kyoungjun/Video-R1/src/r1-v//log/Qwen2.5-VL-7B-Video-7B-newcot70-2f-sft-0612/' \
    --dataset_name "/blob/kyoungjun/Video-R1/src/r1-v/Video-Ours-data/real_gen_r1_grpo_train_w_cot_30.json" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 true \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal false \
    --temporal_gen true \
    --temporal_ver false \
    --len_control true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name Video-R1-Discriminator \
    --save_steps 1000 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model true \
    --num_generations 6  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance