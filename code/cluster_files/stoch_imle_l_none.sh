
#!/bin/bash -l

#$ -N "stoch_imle_l_none"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/stoch_imle_l_none.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/stoch_imle_l_none.err

#$ -t 1-10
#$ -l tmem=12G
#$ -l h_rt=96:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1

test $SGE_TASK_ID -eq 1 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c stoch_imle_l_none -j 1 
test $SGE_TASK_ID -eq 2 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c stoch_imle_l_none -j 2 
test $SGE_TASK_ID -eq 3 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c stoch_imle_l_none -j 3 
test $SGE_TASK_ID -eq 4 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c stoch_imle_l_none -j 4 
test $SGE_TASK_ID -eq 5 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c stoch_imle_l_none -j 5 
test $SGE_TASK_ID -eq 6 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c stoch_imle_l_none -j 6 
test $SGE_TASK_ID -eq 7 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c stoch_imle_l_none -j 7 
test $SGE_TASK_ID -eq 8 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c stoch_imle_l_none -j 8 
test $SGE_TASK_ID -eq 9 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c stoch_imle_l_none -j 9 
test $SGE_TASK_ID -eq 10 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c stoch_imle_l_none -j 10 
