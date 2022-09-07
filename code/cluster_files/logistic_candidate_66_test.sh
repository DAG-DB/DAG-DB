
#!/bin/bash -l

#$ -N "logistic_candidate_66_test"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/logistic_candidate_66_test.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/logistic_candidate_66_test.err

#$ -l tmem=6G
#$ -l h_rt=2:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG

CUDA_LAUNCH_BLOCKING=1

# job_ID that didn't work in search_v1

python3 $HOME/I-MLE-DAG/code/main_dag.py -c logistic_candidate_66_test
