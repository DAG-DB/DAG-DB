
#!/bin/bash -l

#$ -N "search_v1"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/search_v1.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/search_v1.err

#$ -t 1-3
#$ -l tmem=6G
#$ -l h_rt=2:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1

# job_ID that didn't work in search_v1

test $SGE_TASK_ID -eq 1 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c search_v1 -j 84 --lamb 1 --temperature 0.01 --h_null 0 --h_lr 0.1
test $SGE_TASK_ID -eq 2 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c search_v1 -j 319 --lamb 100 --temperature 10 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 3 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c search_v1 -j 320 --lamb 100 --temperature 10 --h_null 1 --h_lr 0.1
