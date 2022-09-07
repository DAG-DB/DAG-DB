
#!/bin/bash -l

#$ -N "search_v0"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/search_v0.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/search_v0.err

#$ -t 1-6
#$ -l tmem=6G
#$ -l h_rt=2:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1

test $SGE_TASK_ID -eq 1 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c search_v0 -j 1 --max_size 54
test $SGE_TASK_ID -eq 2 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c search_v0 -j 2 --max_size 60
test $SGE_TASK_ID -eq 3 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c search_v0 -j 3 --max_size 66
test $SGE_TASK_ID -eq 4 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c search_v0 -j 4 --max_size 72
test $SGE_TASK_ID -eq 5 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c search_v0 -j 5 --max_size 78
test $SGE_TASK_ID -eq 6 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c search_v0 -j 6 --max_size 84
