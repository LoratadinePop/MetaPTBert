#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

# NUM_GPU=4
# PORT_ID=$(expr $RANDOM + 1000)
# export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=7


python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/bert-base-frozen-22 \
    --seed 22 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --learning_rate 5e-3 \
    --frozen \
    --meta_prefix \
    --layer_wise \
    --pre_seq_len 8 \
    --meta_embed_size 512 \
    --layer_embed_size 128 \
    --meta_hidden_size 512 \
    --prefix_hidden_size 512 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
