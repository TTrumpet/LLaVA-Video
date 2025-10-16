#!/bin/bash

# --- STAGE 2: END-TO-END FINE-TUNING ---

# 1. DEFINE YOUR MODELS
LLM_VERSION="Qwen/Qwen2-7B-Instruct" # Base model, used for reference
VISION_MODEL_VERSION="DeepGlint-AI/rice-vit-large-patch14-560"

# 2. DEFINE YOUR DATA
#    Point this to your visual instruction-following dataset (e.g., LLaVA-Instruct-150K)
IMAGE_FOLDER="path/to/your/instruction_data/images"
VIDEO_FOLDER="/datasets/vidor/videos" # Or your video dataset folder
DATA_YAML="path/to/your/stage2_data.yaml"

# 3. DEFINE RUN NAME AND CHECKPOINT FROM STAGE 1
#    KEY CHANGE: This MUST point to the checkpoint saved by the Stage 1 script
STAGE1_CHECKPOINT="./work_dirs/stage1_mvt_qwen2_projector_pretrain/checkpoint-500" # Example path
RUN_NAME="stage2_mvt_qwen2_instruction_tune"

export PYTHONPATH="./:$PYTHONPATH"
/home/ti727611/.conda/envs/llava/bin/deepspeed --master_port 30000 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $STAGE1_CHECKPOINT \
    --vision_tower ${VISION_MODEL_VERSION} \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --mm_tunable_parts "mm_mlp_adapter,mm_language_model" \
    --learning_rate 2e-5 \
    --version "qwen_1_5" \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir ./work_dirs/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb