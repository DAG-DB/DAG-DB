import itertools
import os

import numpy as np
import pandas as pd


FILE_TYPE = 'IMLE_Logistic_None'

assert FILE_TYPE in [
	'IMLE_Logistic_66', 'STE_Logistic_84', 'IMLE_Logistic_None']

ABLATION_LOGS = os.path.join(
	'..', '..', '..', 'logs_to_collate', 'ablation', FILE_TYPE )
N_DAGS = 24
HPS = ['MAX_SIZE', 'z_RHO', 'z_MU']


def get_quantity(section, name, end_name, int_flag=False):
	# numeric = int if int_flag else float
	start = section.index(name) + len(name) + 1
	end = start + section[start: ].index(end_name)
	quantity = section[start: end]
	return quantity


def get_metrics(j, content, results):
	start_hps = content.index('hyperparameters =')
	content = content[start_hps: ]
	hp_value = dict()
	for hp in HPS:
		# if hp == 'MAX_SIZE':
		# 	str_flag = True
		# else:
		# 	str_flag = False
		hp_value[hp] = get_quantity(content, f"'{hp}':", ',')

	content = content.split('epoch=1000')[1:]
	if content == list():
		print('\tNo DAGs for job {j}')
		return results, True
	dag = None
	for dag, section in enumerate(content):
		nshd_c = get_quantity(section, 'nSHD_c', 'nSHD')
		tpr_c = get_quantity(section, 'tpr_c', 'change_')
		size = get_quantity(section, 'size', 'nSHD_c', int_flag=True)
		results['j'].append(j)
		for hp in HPS:
			results[hp].append(hp_value[hp])
		results['dag'].append(dag)
		results['nshd_c'].append(nshd_c)
		results['tpr_c'].append(tpr_c)
		results['size'].append(size)
	if dag + 1 != N_DAGS:
		print(f'\tToo few DAGs on job {j}: only {dag + 1} of {N_DAGS}')
		error_flag = True
	else:
		error_flag = False
	return results, error_flag


def get_file_type_results(file_type):
	results = {
		**{hp: [] for hp in HPS},
		**{'j': [], 'dag': [], 'nshd_c': [], 'tpr_c': [], 'size': []}
	}
	clean_jobs = set()
	error_count = 0
	error_jobs = list()
	for file_name in os.listdir(ABLATION_LOGS):
		if file_type not in file_name:
			continue
		j = int(file_name[-7: -4])
		# if j != 183:
		# 	continue
		with open(os.path.join(ABLATION_LOGS, file_name), 'r') as f:
			content = f.read()
		results, error_flag = get_metrics(j, content, results)
		if error_flag:
			error_count += 1
			error_jobs.append(j)
		else:
			clean_jobs.add(j)

	all_jobs = set(range(1, 8 + 1))
	if clean_jobs == all_jobs:
		print(f'All jobs present for {file_type=}')
	else:
		print(f'Jobs {all_jobs - clean_jobs} missing for {file_type=}')

	error_jobs = np.array(error_jobs)#
	error_jobs.sort()
	print(f'\n\t{error_jobs=}')
	print(f'\t{error_count=}\n')

	return results


def main():
	results = get_file_type_results('ablation')
	results = pd.DataFrame(results)
	results.to_csv(f'{FILE_TYPE}_ablation.csv', index=False)


if __name__ == '__main__':
	main()
