#!/usr/bin/sh

CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run attacks/mist.py \
    --cuda \
    --low_vram_mode \
    --instance_data_dir data/training \
    --output_dir output/ \
    --class_data_dir data/class \
    --instance_prompt "an illustration" \
    --class_prompt "an illustration" \
    --mixed_precision bf16 \
    --seed 1
