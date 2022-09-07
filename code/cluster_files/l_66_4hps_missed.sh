
#!/bin/bash -l

#$ -N "l_66_4hps_missed"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/l_66_4hps_missed.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/l_66_4hps_missed.err

#$ -t 1-66
#$ -l tmem=6G
#$ -l h_rt=6:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1

test $SGE_TASK_ID -eq 1 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 7 --lamb 0.1 --temperature 0.01 --h_null 0.001 --h_lr 0.01
test $SGE_TASK_ID -eq 2 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 17 --lamb 0.1 --temperature 0.01 --h_null 1 --h_lr 0.0001
test $SGE_TASK_ID -eq 3 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 36 --lamb 0.1 --temperature 0.1 --h_null 0.1 --h_lr 0.1
test $SGE_TASK_ID -eq 4 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 39 --lamb 0.1 --temperature 0.1 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 5 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 50 --lamb 0.1 --temperature 1 --h_null 0.01 --h_lr 0.001
test $SGE_TASK_ID -eq 6 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 57 --lamb 0.1 --temperature 1 --h_null 1 --h_lr 0.0001
test $SGE_TASK_ID -eq 7 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 59 --lamb 0.1 --temperature 1 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 8 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 60 --lamb 0.1 --temperature 1 --h_null 1 --h_lr 0.1
test $SGE_TASK_ID -eq 9 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 70 --lamb 0.1 --temperature 10 --h_null 0.01 --h_lr 0.001
test $SGE_TASK_ID -eq 10 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 81 --lamb 1 --temperature 0.01 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 11 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 83 --lamb 1 --temperature 0.01 --h_null 0 --h_lr 0.01
test $SGE_TASK_ID -eq 12 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 85 --lamb 1 --temperature 0.01 --h_null 0.001 --h_lr 0.0001
test $SGE_TASK_ID -eq 13 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 88 --lamb 1 --temperature 0.01 --h_null 0.001 --h_lr 0.1
test $SGE_TASK_ID -eq 14 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 99 --lamb 1 --temperature 0.01 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 15 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 100 --lamb 1 --temperature 0.01 --h_null 1 --h_lr 0.1
test $SGE_TASK_ID -eq 16 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 101 --lamb 1 --temperature 0.1 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 17 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 111 --lamb 1 --temperature 0.1 --h_null 0.01 --h_lr 0.01
test $SGE_TASK_ID -eq 18 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 113 --lamb 1 --temperature 0.1 --h_null 0.1 --h_lr 0.0001
test $SGE_TASK_ID -eq 19 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 114 --lamb 1 --temperature 0.1 --h_null 0.1 --h_lr 0.001
test $SGE_TASK_ID -eq 20 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 118 --lamb 1 --temperature 0.1 --h_null 1 --h_lr 0.001
test $SGE_TASK_ID -eq 21 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 122 --lamb 1 --temperature 1 --h_null 0 --h_lr 0.001
test $SGE_TASK_ID -eq 22 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 124 --lamb 1 --temperature 1 --h_null 0 --h_lr 0.1
test $SGE_TASK_ID -eq 23 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 125 --lamb 1 --temperature 1 --h_null 0.001 --h_lr 0.0001
test $SGE_TASK_ID -eq 24 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 129 --lamb 1 --temperature 1 --h_null 0.01 --h_lr 0.0001
test $SGE_TASK_ID -eq 25 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 130 --lamb 1 --temperature 1 --h_null 0.01 --h_lr 0.001
test $SGE_TASK_ID -eq 26 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 139 --lamb 1 --temperature 1 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 27 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 140 --lamb 1 --temperature 1 --h_null 1 --h_lr 0.1
test $SGE_TASK_ID -eq 28 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 148 --lamb 1 --temperature 10 --h_null 0.001 --h_lr 0.1
test $SGE_TASK_ID -eq 29 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 152 --lamb 1 --temperature 10 --h_null 0.01 --h_lr 0.1
test $SGE_TASK_ID -eq 30 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 153 --lamb 1 --temperature 10 --h_null 0.1 --h_lr 0.0001
test $SGE_TASK_ID -eq 31 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 154 --lamb 1 --temperature 10 --h_null 0.1 --h_lr 0.001
test $SGE_TASK_ID -eq 32 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 155 --lamb 1 --temperature 10 --h_null 0.1 --h_lr 0.01
test $SGE_TASK_ID -eq 33 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 156 --lamb 1 --temperature 10 --h_null 0.1 --h_lr 0.1
test $SGE_TASK_ID -eq 34 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 157 --lamb 1 --temperature 10 --h_null 1 --h_lr 0.0001
test $SGE_TASK_ID -eq 35 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 158 --lamb 1 --temperature 10 --h_null 1 --h_lr 0.001
test $SGE_TASK_ID -eq 36 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 159 --lamb 1 --temperature 10 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 37 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 161 --lamb 10 --temperature 0.01 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 38 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 162 --lamb 10 --temperature 0.01 --h_null 0 --h_lr 0.001
test $SGE_TASK_ID -eq 39 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 163 --lamb 10 --temperature 0.01 --h_null 0 --h_lr 0.01
test $SGE_TASK_ID -eq 40 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 164 --lamb 10 --temperature 0.01 --h_null 0 --h_lr 0.1
test $SGE_TASK_ID -eq 41 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 165 --lamb 10 --temperature 0.01 --h_null 0.001 --h_lr 0.0001
test $SGE_TASK_ID -eq 42 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 166 --lamb 10 --temperature 0.01 --h_null 0.001 --h_lr 0.001
test $SGE_TASK_ID -eq 43 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 167 --lamb 10 --temperature 0.01 --h_null 0.001 --h_lr 0.01
test $SGE_TASK_ID -eq 44 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 173 --lamb 10 --temperature 0.01 --h_null 0.1 --h_lr 0.0001
test $SGE_TASK_ID -eq 45 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 174 --lamb 10 --temperature 0.01 --h_null 0.1 --h_lr 0.001
test $SGE_TASK_ID -eq 46 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 179 --lamb 10 --temperature 0.01 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 47 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 181 --lamb 10 --temperature 0.1 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 48 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 183 --lamb 10 --temperature 0.1 --h_null 0 --h_lr 0.01
test $SGE_TASK_ID -eq 49 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 187 --lamb 10 --temperature 0.1 --h_null 0.001 --h_lr 0.01
test $SGE_TASK_ID -eq 50 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 188 --lamb 10 --temperature 0.1 --h_null 0.001 --h_lr 0.1
test $SGE_TASK_ID -eq 51 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 195 --lamb 10 --temperature 0.1 --h_null 0.1 --h_lr 0.01
test $SGE_TASK_ID -eq 52 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 200 --lamb 10 --temperature 0.1 --h_null 1 --h_lr 0.1
test $SGE_TASK_ID -eq 53 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 202 --lamb 10 --temperature 1 --h_null 0 --h_lr 0.001
test $SGE_TASK_ID -eq 54 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 221 --lamb 10 --temperature 10 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 55 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 223 --lamb 10 --temperature 10 --h_null 0 --h_lr 0.01
test $SGE_TASK_ID -eq 56 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 226 --lamb 10 --temperature 10 --h_null 0.001 --h_lr 0.001
test $SGE_TASK_ID -eq 57 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 255 --lamb 100 --temperature 0.01 --h_null 0.1 --h_lr 0.01
test $SGE_TASK_ID -eq 58 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 256 --lamb 100 --temperature 0.01 --h_null 0.1 --h_lr 0.1
test $SGE_TASK_ID -eq 59 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 261 --lamb 100 --temperature 0.1 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 60 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 266 --lamb 100 --temperature 0.1 --h_null 0.001 --h_lr 0.001
test $SGE_TASK_ID -eq 61 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 282 --lamb 100 --temperature 1 --h_null 0 --h_lr 0.001
test $SGE_TASK_ID -eq 62 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 285 --lamb 100 --temperature 1 --h_null 0.001 --h_lr 0.0001
test $SGE_TASK_ID -eq 63 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 292 --lamb 100 --temperature 1 --h_null 0.01 --h_lr 0.1
test $SGE_TASK_ID -eq 64 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 294 --lamb 100 --temperature 1 --h_null 0.1 --h_lr 0.001
test $SGE_TASK_ID -eq 65 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 306 --lamb 100 --temperature 10 --h_null 0.001 --h_lr 0.001
test $SGE_TASK_ID -eq 66 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 315 --lamb 100 --temperature 10 --h_null 0.1 --h_lr 0.01
