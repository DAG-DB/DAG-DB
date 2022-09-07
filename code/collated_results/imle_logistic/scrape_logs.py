import itertools
import os

import numpy as np
import pandas as pd

LOGS = os.path.join('..', '..', '..', 'logs_to_collate', 'l_66')

N_DAGS = 24
ARGS = 	{'lamb': [0.1, 1, 10, 100],
	'temperature': [0.01, 0.1, 1, 10],
	'h_null': [0, 1e-3, 1e-2, 1e-1, 1],
	'h_lr': [1e-4, 1e-3, 1e-2, 1e-1]
}
HPS = ['LAMBDA', 'NOISE_TEMPERATURE', 'h_NULL', 'h_LR']
FILE_TYPES = ['l_66_4hps', 'l_66none_4hps']


def get_max_j():
	max_j = {file_type: -1 for file_type in FILE_TYPES}
	for file_name in os.listdir(LOGS):
		if FILE_TYPES[0] in file_name:
			file_type = FILE_TYPES[0]
		elif  FILE_TYPES[1] in file_name:
			file_type = FILE_TYPES[1]
		else:
			raise NotImplementedError
		if (j := int(file_name[-7: -4])) > max_j[file_type]:
			max_j[file_type] = j
	return max_j


def get_quantity(section, name, end_name, int_flag=False):
	numeric = int if int_flag else float
	start = section.index(name) + len(name) + 1
	end = start + section[start: ].index(end_name)
	quantity = numeric(section[start: end])
	return quantity


def get_metrics(j, content, results):
	start_hps = content.index('hyperparameters =')
	content = content[start_hps: ]
	hp_value = dict()
	for hp in HPS:
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


def get_file_type_results(file_type, max_j):
	jobs = itertools.product(*ARGS.values())
	assert len(list(jobs)) == (max_j := max_j[file_type])
	results = {
		**{hp: [] for hp in HPS},
		**{'j': [], 'dag': [], 'nshd_c': [], 'tpr_c': [], 'size': []}
	}
	error_count = 0
	error_jobs = list()
	for file_name in os.listdir(LOGS):
		if file_type not in file_name:
			continue
		j = int(file_name[-7: -4])
		# if j != 183:
		# 	continue
		with open(os.path.join(LOGS, file_name), 'r') as f:
			content = f.read()
		results, error_flag = get_metrics(j, content, results)
		if error_flag:
			error_count += 1
			error_jobs.append(j)

	error_jobs = np.array(error_jobs)#
	error_jobs.sort()
	print(f'\n\t{error_jobs=}')
	print(f'\t{error_count=}\n')

	return results


def main():
	max_j = get_max_j()
	for file_type in FILE_TYPES:
		print('\n', file_type)
		results = get_file_type_results(file_type, max_j)
		results = pd.DataFrame(results)
		results.to_csv(file_type + '.csv', index=False)


if __name__ == '__main__':
	main()
