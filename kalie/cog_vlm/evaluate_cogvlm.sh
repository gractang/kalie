#! /bin/bash
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

NUM_GPUS_PER_WORKER=2
MP_SIZE=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
MODEL_TYPE="cogvlm-base-490"   
VERSION="chat"
MODEL_ARGS="--from_pretrained ./checkpoints/merged_model \
    --max_length 3000 \
    --lora_rank 10 \
    --use_lora \
    --local_tokenizer lmsys/vicuna-7b-v1.5 \
    --version $VERSION"
# Tips: If training models of resolution 244, you can set --max_length smaller 

OPTIONS_SAT="SAT_HOME=~/.sat_models"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 LOCAL_WORLD_SIZE=$NUM_GPUS_PER_WORKER"
HOST_FILE_PATH="hostfile"

train_data=$(grep -oP '"TRAIN_PATH":\s*"\K[^"]+' "./kalie/cog_vlm/file_path_config.json")
test_data=$(grep -oP '"VALID_PATH":\s*"\K[^"]+' "./kalie/cog_vlm/file_path_config.json")

gpt_options=" \
       --experiment-name finetune-$MODEL_TYPE \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 0 \
       --resume-dataloader \
       $MODEL_ARGS \
       --train-data ${train_data} \
       --test-data ${test_data} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .02 \
       --checkpoint-activations \
       --save-interval 200 \
       --eval-interval 200 \
       --save "./checkpoints" \
       --strict-eval \
       --eval-batch-size 1 \
       --split 1. \
       --deepspeed_config ./kalie/cog_vlm/test_config_bf16.json \
       --skip-init \
       --seed 2023
"

run_cmd="${OPTIONS_NCCL} ${OPTIONS_SAT} deepspeed --master_port 16001 --hostfile ${HOST_FILE_PATH} ./kalie/cog_vlm/evaluate_cogvlm_demo.py ${gpt_options}"

echo ${run_cmd}
eval ${run_cmd}

set +x
