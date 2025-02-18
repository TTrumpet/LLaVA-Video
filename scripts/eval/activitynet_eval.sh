#!/bin/bash
# CKPT_NAME="llavavideo-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-elysium_100K_lora_a3"
CKPT_NAME="LLaVA-Video-7B-Qwen2"
# model_path="results/${CKPT_NAME}"
model_path="lmms-lab/LLaVA-Video-7B-Qwen2"
GPT_Zero_Shot_QA="/home/jfioresi/vlm/Video-LLaVA/eval/GPT_Zero_Shot_QA"
# video_dir="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/all_test"
video_dir="/groups/mshah/data/activity_net.v1-3/videos"
gt_file_question="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/test_a.json"
output_dir="evaluation/GPT_Zero_Shot_QA/Activitynet_Zero_Shot_QA/${CKPT_NAME}"
model_base=None # "lmms-lab/LLaVA-Video-7B-Qwen2"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} /home/jfioresi/miniforge3/envs/llava/bin/python3 evaluation/run_inference_video_qa_act.py \
      --model_path ${model_path} \
      --model_base ${model_base} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done
