#!/bin/bash
CKPT_NAME="llavavideo-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-elysium_100K_lora_a3"
# CKPT_NAME="llavavideo-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-elysium_100K_lora_32f_a1"
model_path="/lustre/fs1/home/jfioresi/vlm/LLaVA-Video/work_dirs/${CKPT_NAME}"
model_base="lmms-lab/LLaVA-Video-7B-Qwen2"
# model_base="/lustre/fs1/home/jfioresi/vlm/LLaVA-Video/work_dirs/llavavideo-Qwen2-7B_elysium_bbox_merged_a2"
# CKPT_NAME="LLaVA-Video-7B-Qwen2"
# model_path="lmms-lab/LLaVA-Video-7B-Qwen2"
# model_base=None
video_dir="/datasets/"
gt_file_question="/lustre/fs1/home/jfioresi/datasets/elysium_track/conv_ElysiumTrack-20K-Newton.json"


/home/jfioresi/miniforge3/envs/llava/bin/python evaluation/eval_elysium.py \
    --model_path ${model_path} \
    --model_base ${model_base} \
    --video_dir ${video_dir} \
    --gt_file_question ${gt_file_question} \
