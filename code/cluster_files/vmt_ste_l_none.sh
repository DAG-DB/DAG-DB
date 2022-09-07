
#!/bin/bash -l

#$ -N "vmt_ste_l_none"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/vmt_ste_l_none.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/vmt_ste_l_none.err

#$ -t 1-15
#$ -l tmem=12G
#$ -l h_rt=24:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1

test $SGE_TASK_ID -eq 1 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 1 --dags "'010_ER1_gaussian_ev_2022-08-20_BST_11:36:33,306_create_dags'"
test $SGE_TASK_ID -eq 2 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 2 --dags "'030_ER1_gaussian_ev_2022-08-20_BST_11:37:19,132_create_dags'"
test $SGE_TASK_ID -eq 3 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 3 --dags "'010_ER2_gaussian_ev_2022-08-20_BST_11:37:40,257_create_dags'"
test $SGE_TASK_ID -eq 4 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 4 --dags "'030_SF2_gaussian_ev_2022-08-20_BST_11:39:32,564_create_dags'"
test $SGE_TASK_ID -eq 5 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 5 --dags "'030_ER4_gaussian_ev_2022-08-20_BST_11:39:01,675_create_dags'"
test $SGE_TASK_ID -eq 6 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 6 --dags "'100_SF4_gaussian_ev_2022-08-20_BST_11:40:07,367_create_dags'"
test $SGE_TASK_ID -eq 7 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 7 --dags "'010_SF2_gaussian_ev_2022-08-20_BST_11:39:28,507_create_dags'"
test $SGE_TASK_ID -eq 8 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 8 --dags "'100_ER2_gaussian_ev_2022-08-20_BST_11:37:54,238_create_dags'"
test $SGE_TASK_ID -eq 9 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 9 --dags "'100_ER4_gaussian_ev_2022-08-20_BST_11:39:15,764_create_dags'"
test $SGE_TASK_ID -eq 10 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 10 --dags "'100_SF2_gaussian_ev_2022-08-20_BST_11:39:41,987_create_dags'"
test $SGE_TASK_ID -eq 11 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 11 --dags "'030_SF4_gaussian_ev_2022-08-20_BST_11:39:59,723_create_dags'"
test $SGE_TASK_ID -eq 12 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 12 --dags "'030_ER2_gaussian_ev_2022-08-20_BST_11:37:48,933_create_dags'"
test $SGE_TASK_ID -eq 13 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 13 --dags "'010_ER4_gaussian_ev_2022-08-20_BST_11:38:06,145_create_dags'"
test $SGE_TASK_ID -eq 14 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 14 --dags "'010_SF4_gaussian_ev_2022-08-20_BST_11:39:51,306_create_dags'"
test $SGE_TASK_ID -eq 15 && sleep 30 && PYTHONPATH=. python3 $HOME/I-MLE-DAG/code/main_dag.py -c vmt_ste_l_none -j 15 --dags "'100_ER1_gaussian_ev_2022-08-20_BST_11:37:28,139_create_dags'"
