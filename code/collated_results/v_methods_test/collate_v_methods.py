import datetime
import itertools
import os

import numpy as np
from dateutil.tz import tzlocal
import odf  # Used in to_excel
import pandas as pd

from collated_results.imle_logistic.scrape_logs import get_quantity
from utils.metrics import count_accuracy, is_dag


STD_NOW = datetime.datetime.now(tz=tzlocal()).strftime("%Y-%m-%d_%Z_%H:%M:%S,%f")[:-3]
DATA_FOLDER = os.path.join('..', '..', '..', 'data', 'v_methods_test')
RESULTS_FOLDER = os.path.join('..', '..', '..', 'results', 'v_methods_test')
COLLATED_RESULTS = STD_NOW + '_collated_results.ods'
LOGS = os.path.join(RESULTS_FOLDER, 'internal')
N_FILES = 15
N_DAGS = 24
INTERNAL_FILE_TYPES = [
	'vmt_ste_l_none', 'vmt_imle_l_none', 'vmt_ste_l_84', 'vmt_imle_l_66']
VERBOSE = False


def notears_moveable_thresholding(adj_pred):
	"""
	Sets smallest non-zero absolute element o adj_pred to 0 until adj_pred
	is adjacency matrix of DAG

	:param adj_pred: (d, d) float numpy array
	:return: (d, d) int numpy array with binary entries
	"""
	monitor = np.abs(adj_pred) + np.where(adj_pred == 0, np.inf, 0)
	while not is_dag(adj_pred):
		arg_min = np.unravel_index(
			np.argmin(monitor, axis=None), monitor.shape)
		adj_pred[arg_min] = 0
		monitor[arg_min] = np.inf

	adj_pred = adj_pred.astype(bool).astype(int)
	return adj_pred


def process_dag_folder(dag_filepath, dag_folder):
	result = {
		'graph_no': [], 'algo': [], 'nSHD_c': [], 'tpr_c': [], 'size': []}
	for algorithm in os.listdir(RESULTS_FOLDER):
		if algorithm == 'internal':
			continue
		results_path = os.path.join(RESULTS_FOLDER, algorithm, dag_folder)
		for adj_true_file in os.listdir(dag_filepath):
			if ('data' in adj_true_file) or (adj_true_file[-4:] == '.log'):
				continue
			graph_no = int(adj_true_file[: 2])
			if VERBOSE:
				print('\t', dag_filepath, adj_true_file)
			adj_true = np.load(
				os.path.join(dag_filepath, adj_true_file)).astype(
				bool).astype(int)
			adj_pred_file = f'{adj_true_file[:-4]}_pred_{algorithm}.npy'
			# print(adj_pred_file)
			adj_pred_path = os.path.join(results_path, adj_pred_file)
			adj_pred = np.load(adj_pred_path)
			if 'NOTEARS' in algorithm:
				# Should have done this in notears_runs.py
				adj_pred = notears_moveable_thresholding(adj_pred)
			metrics = count_accuracy(adj_true, adj_pred)
			result['graph_no'].append(graph_no)
			result['algo'].append(algorithm)
			result['nSHD_c'].append(metrics['nshd_c'])
			result['tpr_c'].append(metrics['tpr_c'])
			result['size'].append(metrics['pred_size'])
	df = pd.DataFrame.from_dict(result)
	df['graph_type'] = dag_folder[4 : 7]
	df['d'] = int(dag_folder[: 3])
	df['graph_set'] = dag_folder
	return df


def get_max_j():
	max_j = {file_type: -1 for file_type in INTERNAL_FILE_TYPES}
	for file_name in os.listdir(LOGS):
		file_type = None
		for possible_file_type in INTERNAL_FILE_TYPES:
			if possible_file_type in file_name:
				file_type = possible_file_type
				break
		if (j := (int(file_name[-7: -4]) % 100)) > max_j[file_type]:
			max_j[file_type] = j
	return max_j


def get_graph_set(section, name="'DAGS':", delimiter="'"):
	start = section.index(name) + len(name) + 1
	start = section[start:].index(delimiter) + start + 1
	end = start + section[start: ].index(delimiter)
	graph_set = section[start: end]
	return graph_set


def get_metrics_from_log(j, content):
	start_hps = content.index('hyperparameters =')
	graph_set = get_graph_set(content)

	result = {'graph_no': [], 'nSHD_c': [], 'tpr_c': [], 'size': []}
	content = content[start_hps: ]
	content = content.split('epoch=1000')[1:]
	if content == list():
		error_flag = True
		print('\tNo DAGs for job no. {j}')
	else:
		graph_no = None
		for graph_no, section in enumerate(content):
			nshd_c = get_quantity(section, 'nSHD_c', 'nSHD')
			tpr_c = get_quantity(section, 'tpr_c', 'change_')
			size = get_quantity(section, 'size', 'nSHD_c', int_flag=True)
			result['graph_no'].append(graph_no)
			result['nSHD_c'].append(nshd_c)
			result['tpr_c'].append(tpr_c)
			result['size'].append(size)
		if graph_no + 1 != N_DAGS:
			print(f'\tToo few DAGs for job no. {j}: '
				  f'only {graph_no + 1} of {N_DAGS}')
			error_flag = True
		else:
			error_flag = False

	result = pd.DataFrame(result)
	result['graph_set'] = graph_set
	u0 = graph_set.find('_')
	u1 = graph_set[u0 + 1:].find('_') + u0 + 1
	result['d'] = int(graph_set[:u0])
	result['graph_type'] = graph_set[u0 + 1: u1]

	return result, error_flag


def get_file_type_results(file_type):
	df = pd.DataFrame.from_dict(
		{'graph_type': [], 'd': [], 'graph_no': [], 'nSHD_c': [],
		 'tpr_c': [], 'size': [], 'graph_set': []})
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
		file_name_df, error_flag = get_metrics_from_log(j, content)
		df = pd.concat((df, file_name_df), ignore_index=True)
		if error_flag:
			error_count += 1
			error_jobs.append(j)

	error_jobs = np.array(error_jobs)#
	error_jobs.sort()
	print(f'\n\t{error_jobs=}')
	print(f'\t{error_count=}\n')

	df['algo'] = file_type
	return df


def main():
	collated = pd.DataFrame.from_dict(
		{'graph_type': [], 'd': [], 'graph_no': [], 'algo': [], 'nSHD_c': [],
		 'tpr_c': [], 'size': [], 'graph_set': []})

	# Collate internal results
	max_j = get_max_j()
	for file_type in INTERNAL_FILE_TYPES:
		print('\n', file_type)
		assert max_j[file_type] == N_FILES
		df = get_file_type_results(file_type)
		collated = pd.concat((collated, df), ignore_index=True)

	# Collate external results
	for dag_folder in os.listdir(DATA_FOLDER):
		dag_filepath = os.path.join(DATA_FOLDER, dag_folder)
		if os.path.isdir(dag_filepath):
			print(f'\n Processing data {dag_folder}')
			df = process_dag_folder(dag_filepath, dag_folder)
			collated = pd.concat((collated, df), ignore_index=True)

	print(collated)
	collated.to_excel(COLLATED_RESULTS, engine='odf')


def file_check(file_type):
	print(f'\n{file_type}')
	gt_d = [(gt, d) for gt in ['ER1', 'ER2', 'ER4', 'SF2', 'SF4']
			 for d in [10, 30, 100]]
	gt_d = {elt: [] for elt in gt_d}
	for file_name in os.listdir(LOGS):
		if file_type not in file_name:
			continue
		j = file_name[-7: -4]
		# if j != 183:
		# 	continue
		with open(os.path.join(LOGS, file_name), 'r') as f:
			content = f.read()
		file_name_df, error_flag = get_metrics_from_log(j, content)
		graph_type = file_name_df.iloc[0]['graph_type']
		d = file_name_df.iloc[0]['d']
		gt_d[(graph_type, d)].append(j)
	# print(gt_d)

	for key, value in gt_d.items():
		if len(value) != 1:
			print(f'{key}: {value}')


if __name__ == '__main__':
	main()
	# [file_check(file_type) for file_type in INTERNAL_FILE_TYPES]

