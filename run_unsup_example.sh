#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

# NUM_GPU=4
# PORT_ID=$(expr $RANDOM + 1000)
# export OMP_NUM_THREADS=8


# python train.py \
#     --model_name_or_path bert-base-uncased \
#     --train_file data/wiki1m_for_simcse.txt \
#     --output_dir result/bert-base-upsup-SimCSE \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 64 \
#     --learning_rate 3e-5 \
#     --max_seq_length 32 \
#     --evaluation_strategy steps \
#     --metric_for_best_model stsb_spearman \
#     --load_best_model_at_end \
#     --eval_steps 125 \
#     --pooler_type cls \
#     --mlp_only_train \
#     --overwrite_output_dir \
#     --temp 0.05 \
#     --do_train \
#     --do_eval \
#     --fp16 \
#     "$@"

export CUDA_VISIBLE_DEVICES=0

python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/bert-base-hyper-layer-frozen-epoch2-lr_1e-4-prelen_8-seed42 \
    --seed 42 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 128 \
    --learning_rate 1e-4 \
    --frozen \
    --hyper_prefix \
    --layer_wise \
    --pre_seq_len 8 \
    --meta_embed_size 32 \
    --layer_embed_size 32 \
    --meta_hidden_size 32 \
    --prefix_hidden_size 64 \
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
