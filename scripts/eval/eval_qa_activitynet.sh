

GPT_Zero_Shot_QA="evaluation/GPT_Zero_Shot_QA"
output_name="LLaVA-Video-7B-Qwen2"
pred_path="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/${output_name}/merge.jsonl"
# output_dir="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/${output_name}/mixtral_8x7B"
output_dir="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/${output_name}/llama3.1_8B"
output_json="${GPT_Zero_Shot_QA}/Activitynet_Zero_Shot_QA/${output_name}/results.json"
# model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"
# model_name="mistralai/Mistral-7B-Instruct-v0.2"
model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"


/lustre/fs1/home/jfioresi/miniforge3/envs/vllm/bin/python3 evaluation/llama_eval_video_qa.py \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --model_name ${model_name}
