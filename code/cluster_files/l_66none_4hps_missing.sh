
#!/bin/bash -l

#$ -N "l_66none_4hps_missing"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/l_66none_4hps_missing.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/l_66none_4hps_missing.err

$ -t 1-320
#$ -l tmem=6G
#$ -l h_rt=2:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1

python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 1 --lamb 100. --temperature 0.01 --h_null 1.0 --h_lr 0.00001