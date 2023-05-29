cmd_file=$1
srun --gres=gpu:4 --cpus-per-task=64 -N 4 --mem=1T --time 24:00:00 --qos=sched_level_2    --pty bash $cmd_file
