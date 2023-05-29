torchrun --nproc_per_node 4 \
    --master_addr 0.0.0.0 \
    train.py \
    --cfg-path train_configs/minigpt4_stage2_finetune.yaml