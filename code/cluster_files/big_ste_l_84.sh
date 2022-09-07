
#!/bin/bash -l

#$ -N "big_ste_l_84"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/big_ste_l_84.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/big_ste_l_84.err

#$ -t 1-10
#$ -l tmem=16G
#$ -l h_rt=96:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1

test $SGE_TASK_ID -eq 1 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c big_ste_l_84 -j 1 --dags "'000'"
test $SGE_TASK_ID -eq 2 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c big_ste_l_84 -j 2 --dags "'001'"
test $SGE_TASK_ID -eq 3 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c big_ste_l_84 -j 3 --dags "'002'"
test $SGE_TASK_ID -eq 4 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c big_ste_l_84 -j 4 --dags "'003'"
test $SGE_TASK_ID -eq 5 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c big_ste_l_84 -j 5 --dags "'004'"
test $SGE_TASK_ID -eq 6 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c big_ste_l_84 -j 6 --dags "'005'"
test $SGE_TASK_ID -eq 7 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c big_ste_l_84 -j 7 --dags "'006'"
test $SGE_TASK_ID -eq 8 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c big_ste_l_84 -j 8 --dags "'007'"
test $SGE_TASK_ID -eq 9 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c big_ste_l_84 -j 9 --dags "'008'"
test $SGE_TASK_ID -eq 10 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c big_ste_l_84 -j 10 --dags "'009'"
