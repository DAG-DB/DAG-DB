
#!/bin/bash -l

#$ -N "l_66none_4hps_missed"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/l_66none_4hps_missed.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/l_66none_4hps_missed.err

#$ -t 1-31
#$ -l tmem=6G
#$ -l h_rt=6:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1

test $SGE_TASK_ID -eq 1 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 33 --lamb 0.1 --temperature 0.1 --h_null 0.1 --h_lr 0.0001
test $SGE_TASK_ID -eq 2 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 37 --lamb 0.1 --temperature 0.1 --h_null 1 --h_lr 0.0001
test $SGE_TASK_ID -eq 3 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 89 --lamb 1 --temperature 0.01 --h_null 0.01 --h_lr 0.0001
test $SGE_TASK_ID -eq 4 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 119 --lamb 1 --temperature 0.1 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 5 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 132 --lamb 1 --temperature 1 --h_null 0.01 --h_lr 0.1
test $SGE_TASK_ID -eq 6 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 135 --lamb 1 --temperature 1 --h_null 0.1 --h_lr 0.01
test $SGE_TASK_ID -eq 7 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 139 --lamb 1 --temperature 1 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 8 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 146 --lamb 1 --temperature 10 --h_null 0.001 --h_lr 0.001
test $SGE_TASK_ID -eq 9 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 150 --lamb 1 --temperature 10 --h_null 0.01 --h_lr 0.001
test $SGE_TASK_ID -eq 10 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 152 --lamb 1 --temperature 10 --h_null 0.01 --h_lr 0.1
test $SGE_TASK_ID -eq 11 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 158 --lamb 1 --temperature 10 --h_null 1 --h_lr 0.001
test $SGE_TASK_ID -eq 12 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 159 --lamb 1 --temperature 10 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 13 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 169 --lamb 10 --temperature 0.01 --h_null 0.01 --h_lr 0.0001
test $SGE_TASK_ID -eq 14 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 172 --lamb 10 --temperature 0.01 --h_null 0.01 --h_lr 0.1
test $SGE_TASK_ID -eq 15 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 175 --lamb 10 --temperature 0.01 --h_null 0.1 --h_lr 0.01
test $SGE_TASK_ID -eq 16 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 176 --lamb 10 --temperature 0.01 --h_null 0.1 --h_lr 0.1
test $SGE_TASK_ID -eq 17 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 181 --lamb 10 --temperature 0.1 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 18 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 185 --lamb 10 --temperature 0.1 --h_null 0.001 --h_lr 0.0001
test $SGE_TASK_ID -eq 19 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 206 --lamb 10 --temperature 1 --h_null 0.001 --h_lr 0.001
test $SGE_TASK_ID -eq 20 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 208 --lamb 10 --temperature 1 --h_null 0.001 --h_lr 0.1
test $SGE_TASK_ID -eq 21 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 210 --lamb 10 --temperature 1 --h_null 0.01 --h_lr 0.001
test $SGE_TASK_ID -eq 22 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 226 --lamb 10 --temperature 10 --h_null 0.001 --h_lr 0.001
test $SGE_TASK_ID -eq 23 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 228 --lamb 10 --temperature 10 --h_null 0.001 --h_lr 0.1
test $SGE_TASK_ID -eq 24 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 234 --lamb 10 --temperature 10 --h_null 0.1 --h_lr 0.001
test $SGE_TASK_ID -eq 25 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 240 --lamb 10 --temperature 10 --h_null 1 --h_lr 0.1
test $SGE_TASK_ID -eq 26 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 246 --lamb 100 --temperature 0.01 --h_null 0.001 --h_lr 0.001
test $SGE_TASK_ID -eq 27 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 257 --lamb 100 --temperature 0.01 --h_null 1 --h_lr 0.0001
test $SGE_TASK_ID -eq 28 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 269 --lamb 100 --temperature 0.1 --h_null 0.01 --h_lr 0.0001
test $SGE_TASK_ID -eq 29 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 277 --lamb 100 --temperature 0.1 --h_null 1 --h_lr 0.0001
test $SGE_TASK_ID -eq 30 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 282 --lamb 100 --temperature 1 --h_null 0 --h_lr 0.001
test $SGE_TASK_ID -eq 31 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66none_4hps -j 286 --lamb 100 --temperature 1 --h_null 0.001 --h_lr 0.001
