
#!/bin/bash -l

#$ -N "l_66_4hps"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/l_66_4hps.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/l_66_4hps.err

#$ -t 1-66
#$ -l tmem=6G
#$ -l h_rt=2:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1

test $SGE_TASK_ID -eq 7 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 7 --lamb 0.1 --temperature 0.01 --h_null 0.001 --h_lr 0.01
test $SGE_TASK_ID -eq 17 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 17 --lamb 0.1 --temperature 0.01 --h_null 1 --h_lr 0.0001
test $SGE_TASK_ID -eq 36 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 36 --lamb 0.1 --temperature 0.1 --h_null 0.1 --h_lr 0.1
test $SGE_TASK_ID -eq 39 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 39 --lamb 0.1 --temperature 0.1 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 50 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 50 --lamb 0.1 --temperature 1 --h_null 0.01 --h_lr 0.001
test $SGE_TASK_ID -eq 57 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 57 --lamb 0.1 --temperature 1 --h_null 1 --h_lr 0.0001
test $SGE_TASK_ID -eq 59 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 59 --lamb 0.1 --temperature 1 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 60 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 60 --lamb 0.1 --temperature 1 --h_null 1 --h_lr 0.1
test $SGE_TASK_ID -eq 70 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 70 --lamb 0.1 --temperature 10 --h_null 0.01 --h_lr 0.001
test $SGE_TASK_ID -eq 81 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 81 --lamb 1 --temperature 0.01 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 83 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 83 --lamb 1 --temperature 0.01 --h_null 0 --h_lr 0.01
test $SGE_TASK_ID -eq 85 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 85 --lamb 1 --temperature 0.01 --h_null 0.001 --h_lr 0.0001
test $SGE_TASK_ID -eq 88 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 88 --lamb 1 --temperature 0.01 --h_null 0.001 --h_lr 0.1
test $SGE_TASK_ID -eq 99 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 99 --lamb 1 --temperature 0.01 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 100 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 100 --lamb 1 --temperature 0.01 --h_null 1 --h_lr 0.1
test $SGE_TASK_ID -eq 101 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 101 --lamb 1 --temperature 0.1 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 111 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 111 --lamb 1 --temperature 0.1 --h_null 0.01 --h_lr 0.01
test $SGE_TASK_ID -eq 113 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 113 --lamb 1 --temperature 0.1 --h_null 0.1 --h_lr 0.0001
test $SGE_TASK_ID -eq 114 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 114 --lamb 1 --temperature 0.1 --h_null 0.1 --h_lr 0.001
test $SGE_TASK_ID -eq 118 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 118 --lamb 1 --temperature 0.1 --h_null 1 --h_lr 0.001
test $SGE_TASK_ID -eq 122 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 122 --lamb 1 --temperature 1 --h_null 0 --h_lr 0.001
test $SGE_TASK_ID -eq 124 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 124 --lamb 1 --temperature 1 --h_null 0 --h_lr 0.1
test $SGE_TASK_ID -eq 125 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 125 --lamb 1 --temperature 1 --h_null 0.001 --h_lr 0.0001
test $SGE_TASK_ID -eq 129 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 129 --lamb 1 --temperature 1 --h_null 0.01 --h_lr 0.0001
test $SGE_TASK_ID -eq 130 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 130 --lamb 1 --temperature 1 --h_null 0.01 --h_lr 0.001
test $SGE_TASK_ID -eq 139 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 139 --lamb 1 --temperature 1 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 140 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 140 --lamb 1 --temperature 1 --h_null 1 --h_lr 0.1
test $SGE_TASK_ID -eq 148 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 148 --lamb 1 --temperature 10 --h_null 0.001 --h_lr 0.1
test $SGE_TASK_ID -eq 152 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 152 --lamb 1 --temperature 10 --h_null 0.01 --h_lr 0.1
test $SGE_TASK_ID -eq 153 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 153 --lamb 1 --temperature 10 --h_null 0.1 --h_lr 0.0001
test $SGE_TASK_ID -eq 154 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 154 --lamb 1 --temperature 10 --h_null 0.1 --h_lr 0.001
test $SGE_TASK_ID -eq 155 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 155 --lamb 1 --temperature 10 --h_null 0.1 --h_lr 0.01
test $SGE_TASK_ID -eq 156 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 156 --lamb 1 --temperature 10 --h_null 0.1 --h_lr 0.1
test $SGE_TASK_ID -eq 157 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 157 --lamb 1 --temperature 10 --h_null 1 --h_lr 0.0001
test $SGE_TASK_ID -eq 158 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 158 --lamb 1 --temperature 10 --h_null 1 --h_lr 0.001
test $SGE_TASK_ID -eq 159 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 159 --lamb 1 --temperature 10 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 161 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 161 --lamb 10 --temperature 0.01 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 162 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 162 --lamb 10 --temperature 0.01 --h_null 0 --h_lr 0.001
test $SGE_TASK_ID -eq 163 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 163 --lamb 10 --temperature 0.01 --h_null 0 --h_lr 0.01
test $SGE_TASK_ID -eq 164 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 164 --lamb 10 --temperature 0.01 --h_null 0 --h_lr 0.1
test $SGE_TASK_ID -eq 165 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 165 --lamb 10 --temperature 0.01 --h_null 0.001 --h_lr 0.0001
test $SGE_TASK_ID -eq 166 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 166 --lamb 10 --temperature 0.01 --h_null 0.001 --h_lr 0.001
test $SGE_TASK_ID -eq 167 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 167 --lamb 10 --temperature 0.01 --h_null 0.001 --h_lr 0.01
test $SGE_TASK_ID -eq 173 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 173 --lamb 10 --temperature 0.01 --h_null 0.1 --h_lr 0.0001
test $SGE_TASK_ID -eq 174 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 174 --lamb 10 --temperature 0.01 --h_null 0.1 --h_lr 0.001
test $SGE_TASK_ID -eq 179 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 179 --lamb 10 --temperature 0.01 --h_null 1 --h_lr 0.01
test $SGE_TASK_ID -eq 181 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 181 --lamb 10 --temperature 0.1 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 183 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 183 --lamb 10 --temperature 0.1 --h_null 0 --h_lr 0.01
test $SGE_TASK_ID -eq 187 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 187 --lamb 10 --temperature 0.1 --h_null 0.001 --h_lr 0.01
test $SGE_TASK_ID -eq 188 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 188 --lamb 10 --temperature 0.1 --h_null 0.001 --h_lr 0.1
test $SGE_TASK_ID -eq 195 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 195 --lamb 10 --temperature 0.1 --h_null 0.1 --h_lr 0.01
test $SGE_TASK_ID -eq 200 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 200 --lamb 10 --temperature 0.1 --h_null 1 --h_lr 0.1
test $SGE_TASK_ID -eq 202 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 202 --lamb 10 --temperature 1 --h_null 0 --h_lr 0.001
test $SGE_TASK_ID -eq 221 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 221 --lamb 10 --temperature 10 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 223 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 223 --lamb 10 --temperature 10 --h_null 0 --h_lr 0.01
test $SGE_TASK_ID -eq 226 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 226 --lamb 10 --temperature 10 --h_null 0.001 --h_lr 0.001
test $SGE_TASK_ID -eq 255 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 255 --lamb 100 --temperature 0.01 --h_null 0.1 --h_lr 0.01
test $SGE_TASK_ID -eq 256 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 256 --lamb 100 --temperature 0.01 --h_null 0.1 --h_lr 0.1
test $SGE_TASK_ID -eq 261 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 261 --lamb 100 --temperature 0.1 --h_null 0 --h_lr 0.0001
test $SGE_TASK_ID -eq 266 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 266 --lamb 100 --temperature 0.1 --h_null 0.001 --h_lr 0.001
test $SGE_TASK_ID -eq 282 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 282 --lamb 100 --temperature 1 --h_null 0 --h_lr 0.001
test $SGE_TASK_ID -eq 285 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 285 --lamb 100 --temperature 1 --h_null 0.001 --h_lr 0.0001
test $SGE_TASK_ID -eq 292 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 292 --lamb 100 --temperature 1 --h_null 0.01 --h_lr 0.1
test $SGE_TASK_ID -eq 294 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 294 --lamb 100 --temperature 1 --h_null 0.1 --h_lr 0.001
test $SGE_TASK_ID -eq 306 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 306 --lamb 100 --temperature 10 --h_null 0.001 --h_lr 0.001
test $SGE_TASK_ID -eq 315 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c l_66_4hps -j 315 --lamb 100 --temperature 10 --h_null 0.1 --h_lr 0.01
