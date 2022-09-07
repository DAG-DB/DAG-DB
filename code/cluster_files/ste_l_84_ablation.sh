
#!/bin/bash -l

#$ -N "ste_l_84_ablation"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/ste_l_84_ablation.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/ste_l_84_ablation.err

#$ -t 1-8
#$ -l tmem=16G
#$ -l h_rt=96:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1

test $SGE_TASK_ID -eq 1 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c ste_l_84_ablation -j 1 --max_size None --rho 0 --mu 0
test $SGE_TASK_ID -eq 2 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c ste_l_84_ablation -j 2 --max_size None --rho 0 --mu 0.0102343788252807
test $SGE_TASK_ID -eq 3 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c ste_l_84_ablation -j 3 --max_size None --rho 0.410122763411552 --mu 0
test $SGE_TASK_ID -eq 4 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c ste_l_84_ablation -j 4 --max_size None --rho 0.410122763411552 --mu 0.0102343788252807
test $SGE_TASK_ID -eq 5 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c ste_l_84_ablation -j 5 --max_size 66 --rho 0 --mu 0
test $SGE_TASK_ID -eq 6 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c ste_l_84_ablation -j 6 --max_size 66 --rho 0 --mu 0.0102343788252807
test $SGE_TASK_ID -eq 7 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c ste_l_84_ablation -j 7 --max_size 66 --rho 0.410122763411552 --mu 0
test $SGE_TASK_ID -eq 8 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c ste_l_84_ablation -j 8 --max_size 66 --rho 0.410122763411552 --mu 0.0102343788252807
