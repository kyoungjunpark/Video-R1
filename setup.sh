# Install the packages in r1-v .
cd src/r1-v 
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation
pip install peft
pip install openpyxl

# vLLM support 
pip install vllm==0.7.2

# fix transformers version
# pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef