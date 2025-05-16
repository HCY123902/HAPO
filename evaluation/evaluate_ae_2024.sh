MODEL=model_name_or_path
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,max_num_batched_tokens=32768,gpu_memory_utilization=0.95,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
TASK=aime24
OUTPUT_DIR=evaluation/ae_2024

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --system-prompt="Please reason step by step, and put your final answer within \\boxed{}." \
    --save-details \
    --output-dir $OUTPUT_DIR 