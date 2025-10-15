export PYTHONPATH="./":$PYTHONPATH

# predicte results from inference
#RESULT_DIR=eval_output/LLaVA-ST-Qwen2-7B/llava_st_qwen2_7b
#RESULT_DIR=eval_output/llava_st_qwen2_7b
RESULT_DIR=eval_output/checkpoint-10000

# eval rec, tvg, st-align
python inference/multi_task_eval.py  --result_dir ${RESULT_DIR}
