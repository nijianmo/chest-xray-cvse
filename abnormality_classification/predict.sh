#export DATA_DIR='.'
export DATA_DIR='./mimic'
export TASK_NAME=MRG
export MODEL_NAME='MRG_02_09_2020-17:19:56'

python run.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_test \
    --max_seq_length 48 \
    --do_lower_case \
    --data_dir $DATA_DIR \
    --per_gpu_eval_batch_size=128   \
    --per_gpu_train_batch_size=128   \
    --learning_rate 2e-5 \
    --num_train_epochs 5.0 \
    --save_steps 5000 \
    --output_dir tmp/${MODEL_NAME}
