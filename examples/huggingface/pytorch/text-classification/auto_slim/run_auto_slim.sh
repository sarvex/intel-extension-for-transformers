python3 examples/huggingface/pytorch/text-classification/auto_slim/run_bge_sts_autoslim.py \
    --model_name_or_path "./bge-small-stsb-dense" \
    --task_name "stsb" \
    --max_length 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --distill_loss_weight 2.0 \
    --num_train_epochs 13 \
    --weight_decay 1e-7   \
    --cooldown_epochs 3 \
    --sparsity_warm_epochs 1 \
    --lr_scheduler_type "linear" \
    --seed 10 \
    --do_prune \
    --pruning_frequency 500 \
    --prune_ffn2_sparsity 0.80 \
    --prune_mha_sparsity 0.60 \
    --auto_slim \
    --keyword bgesmall_sts_autoslim_lr${lr}_seed${seed} \
    2>&1 | tee bgesmall_sts_autoslim_lr${lr}_seed${seed}_mha60.log