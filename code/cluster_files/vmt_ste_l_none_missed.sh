
#!/bin/bash -l

#$ -N "vmt_ste_l_none"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/vmt_ste_l_none.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/vmt_ste_l_none.err

#$ -t 1-4
#$ -l tmem=10G
#$ -l h_rt=48:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1


test $SGE_TASK_ID -eq 1 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 103 --dags "'010_ER2_gaussian_ev_2022-08-20_BST_11:37:40,257_create_dags'"
test $SGE_TASK_ID -eq 2 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 105 --dags "'030_ER4_gaussian_ev_2022-08-20_BST_11:39:01,675_create_dags'"
test $SGE_TASK_ID -eq 3 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 106 --dags "'100_SF4_gaussian_ev_2022-08-20_BST_11:40:07,367_create_dags'"
test $SGE_TASK_ID -eq 4 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 109 --dags "'100_ER4_gaussian_ev_2022-08-20_BST_11:39:15,764_create_dags'"
