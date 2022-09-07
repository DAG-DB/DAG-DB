""" Write qsub shell scripts and put in code/experiments/
"""


import itertools
import os

import numpy as np

import __main__


CLUSTER_FOLDER = os.path.join('..', 'cluster_files')


def write_script(
		code_main, config, arg_hp, hrs=2, run_jobs=None, mem=6,
		multiple_runs=None):
	"""
	Write qsub shell scripts and put in code/cluster_files/

	:param code_main: str, name of main file to run: always `main_dag.py' in
	practice
	:param config: str, config file name to use (without .py extension)
	:param arg_hp: dict, hyperparameters to set args for.  Each dict value
	is a list of values to iterate over
	:param hrs: int, how many hours to run fro
	:param run_jobs: None, or list of job numbers, as formed from arg_hp, to
	run.  If none, auto-generated.
	:param mem: int, how much memory to request in GB
	:param multiple_runs: None or int.  If None, ignore; if int run that
	number of identical jobs
	"""
	hrs = max(2, hrs)
	assert int(hrs) == hrs
	file_name = os.path.basename(__main__.__file__)[: -3]
	shell_file = os.path.join(CLUSTER_FOLDER, file_name + '.sh')
	config = file_name if config is None else config

	if multiple_runs is None:
		n_jobs = int(np.prod([len(value) for value in arg_hp.values()])) if \
			run_jobs is None else len(run_jobs)
		job_list = itertools.product(*arg_hp.values())
	else:
		n_jobs = multiple_runs
		job_list = [[]] * n_jobs
	array_line = '' if n_jobs <= 1 else f'#$ -t 1-{n_jobs}'

	shell_script = f"""
#!/bin/bash -l

#$ -N "{file_name}"
#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/I-MLE-DAG/cluster/cluster_logs/{file_name}.out
#$ -e $HOME/I-MLE-DAG/cluster/cluster_logs/{file_name}.err

{array_line}
#$ -l tmem={mem}G
#$ -l h_rt={hrs}:00:00
#$ -l gpu=true

hostname
date

# Activate my environment
conda activate I-MLE-DAG  # Name of conda environment in cluster

CUDA_LAUNCH_BLOCKING=1

"""
	if n_jobs <= 1:
		shell_script += f'PYTHONPATH=. python3 {code_main} -c {config} ' \
						f'> output.log 2>&1'
	else:
		array_count = 0
		for job_id, arg_values in enumerate(job_list, 1):
			if (run_jobs is not None) and (job_id not in run_jobs):
				continue
			array_count += 1
			cl_args = ' '.join([
				f'--{key} {value}' for key, value in
				zip(arg_hp.keys(), arg_values)])
			shell_script += f'test $SGE_TASK_ID -eq {array_count} && sleep 30 && ' \
							f'PYTHONPATH=. python3 {code_main} -c {config} ' \
							f'-j {job_id} {cl_args}\n'
							# f' > output_{str(job_id).zfill(3)}.log 2>&1\n'

	with open(shell_file, 'w') as f:
		f.write(shell_script)

	print(f'Script written to {shell_file}')
