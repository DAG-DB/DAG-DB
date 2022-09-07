
#!/bin/bash -l

#$ -N "vmt_imle_l_none"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/vmt_imle_l_none.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/vmt_imle_l_none.err

#$ -t 1-3
#$ -l tmem=10G
#$ -l h_rt=48:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1

test $SGE_TASK_ID -eq 1 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_imle_l_none -j 101 --dags "'100_ER2_gaussian_ev_2022-08-20_BST_11:37:54,238_create_dags'"
test $SGE_TASK_ID -eq 2 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_imle_l_none -j 102 --dags "'100_SF2_gaussian_ev_2022-08-20_BST_11:39:41,987_create_dags'"
test $SGE_TASK_ID -eq 3 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_imle_l_none -j 106 --dags "'030_ER1_gaussian_ev_2022-08-20_BST_11:37:19,132_create_dags'"
