source /nobackup/users/zitian/code/Heaplax/anaconda3/bin/activate && conda activate && conda activate minigpt4
NODE_RANK=${SLURM_PROCID}
ip2=node${SLURM_NODELIST:5:4}
NODE_LENGTH=${#SLURM_NODELIST}
if [[ ${NODE_LENGTH} == 8  ]]; then
    ip2=node${SLURM_NODELIST:4:4}
else
    ip2=node${SLURM_NODELIST:5:4}
fi
echo $ip2
echo $NODE_RANK
echo $SLURM_JOB_NUM_NODES
torchrun --nproc_per_node=4 \
    train.py \
    --cfg-path train_configs/minigpt4_clevr_finetune.yaml
    #--master_addr ${ip2} \
    #--node_rank ${NODE_RANK} \
    #--nnodes $SLURM_JOB_NUM_NODES \