#!/bin/bash
module load cuda/cuda-11.8.0

# Set up the data folders
IMAGE_FOLDER="X"
VIDEO_FOLDER="/groups/mshah/data/TVQA_Plus"
DATA_YAML="scripts/data/tvqa_plus.yaml" # e.g exp.yaml

nvidia-smi

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Stage 2
PROMPT_VERSION="qwen_1_5"
MID_RUN_NAME="llavavideo-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-TVQA_Plus_lora__a1"
PREV_STAGE_CHECKPOINT="lmms-lab/LLaVA-Video-7B-Qwen2"
#PREV_STAGE_CHECKPOINT="work_dirs/llavavideo-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-elysium_100K_lora/checkpoint-500"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

    # --tune_mm_mlp_adapter True \
    # --lora_enable True --lora_r 128 --lora_alpha 256 \
#/home/jfioresi/miniforge3/envs/llava/bin/deepspeed --master_port 30005 \
    deepspeed --master_port 30005 \
    llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path $DATA_YAML \
    --image_folder $IMAGE_FOLDER \
    --video_folder $VIDEO_FOLDER \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr 2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir ./work_dirs/$MID_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 8 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2
exit 0;
