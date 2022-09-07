""" Place in the src directory of https://github.com/ignavierng/golem,
main branch, commit 9e99e615c43a288532c319a29ac4bc2a0830226a 

Also place the DAG-DB file code/data_management/get_real_data.py in the
Golem src directory.

Follow README instructions for installation of Golem (using Python 3.7), and
also install causaldag and pandas as in the I-MLE-DAG
environment.yml file.

Then place synthetic and/or real data in DATA_FOLDER (see file path below).
"""


import os

import numpy as np

from get_real_data import get_sachs_data
from golem import golem
from utils.train import postprocess
from utils.utils import count_accuracy


ALGORITHM = 'GOLEM-NV'
assert ALGORITHM in ['GOLEM-EV', 'GOLEM-NV']
REAL_DATA = 'syntren'  # = None for synthetic data


# As recommended by GOLEM paper:
SETTINGS = {
	'GOLEM-EV': {'lambda_1': 2e-2, 'lambda_2': 5.0, 'equal_variances': True},
	'GOLEM-NV': {'lambda_1': 2e-3, 'lambda_2': 5.0, 'equal_variances': False},
}
FILE_LABEL = ALGORITHM


DAG_FOLDER = os.path.join('..', 'i_mle_dag')
DATA_FOLDER = os.path.join(DAG_FOLDER, 'data')
RESULTS_SUPER_FOLDER = os.path.join(DAG_FOLDER, 'results')
SACHS_RESULTS_FOLDER = os.path.join(RESULTS_SUPER_FOLDER, 'sachs')
SYNTREN_DATA_FOLDER = os.path.join('..', 'data', 'syntren')
SYNTREN_RESULTS_OUTER_FOLDER = os.path.join(RESULTS_SUPER_FOLDER, 'syntren')
SYNTREN_RESULTS_FOLDER = os.path.join(SYNTREN_RESULTS_OUTER_FOLDER, FILE_LABEL)
RESULTS_FOLDER = os.path.join(RESULTS_SUPER_FOLDER, FILE_LABEL)
for folder in [
	RESULTS_SUPER_FOLDER, SACHS_RESULTS_FOLDER, RESULTS_FOLDER,
	SYNTREN_RESULTS_OUTER_FOLDER, SYNTREN_RESULTS_FOLDER
]:
	if not os.path.exists(folder):
		os.mkdir(folder)


def load_dag_dataset(folder_name, i):
	file_stem = os.path.join(folder_name, f'{str(i).zfill(2)}_')
	X = np.load(file_stem + 'data.npy')
	adj_true = np.load(file_stem + 'adj.npy')
	return X ,adj_true


def train_on_data(X, adj_true, intermediate=False):
	# Even if GOLEM-NV, need the GOLEM-EV to initialise it
	adj_pred = golem(
		X,
		SETTINGS['GOLEM-EV']['lambda_1'],
		SETTINGS['GOLEM-EV']['lambda_2'],
		equal_variances=SETTINGS['GOLEM-EV']['equal_variances'],
		checkpoint_iter=5000
	)
	if ALGORITHM == 'GOLEM-NV':
		if intermediate:
			adj_pred_ev = postprocess(adj_pred, graph_thres=0.3)
			adj_pred_ev = adj_pred_ev.astype(bool).astype(int)
			metrics = count_accuracy(adj_true, adj_pred_ev)
			print('\tIntermediate EV:')
			print(f'\t\tshd={metrics["shd"]}\ttpr={metrics["tpr"]}\t'
				  f'pred_size={metrics["pred_size"]}')
		adj_pred = golem(
			X,
			SETTINGS['GOLEM-NV']['lambda_1'],
			SETTINGS['GOLEM-NV']['lambda_2'],
			equal_variances=SETTINGS['GOLEM-NV']['equal_variances'],
			checkpoint_iter=5000,
			B_init=adj_pred
		)
	# Post-process estimated solution and compute results
	adj_pred = postprocess(adj_pred, graph_thres=0.3)
	adj_pred = adj_pred.astype(bool).astype(int)
	metrics = count_accuracy(adj_true, adj_pred)
	print(f'shd={metrics["shd"]}\ttpr={metrics["tpr"]}\t'
		  f'pred_size={metrics["pred_size"]}')
	return adj_pred


def process_dag_data(dir_path, dir_name, results_sub_folder=None):
	results_sub_folder = os.path.join(RESULTS_FOLDER, dir_name) \
		if results_sub_folder is None else results_sub_folder
	if not os.path.exists(results_sub_folder):
		os.mkdir(results_sub_folder)

	max_i = -1
	for filename in os.listdir(dir_path):
		if not filename.endswith('_data.npy') or (max_i >= int(filename[:2])):
			continue
		max_i = int(filename[:2])
	dataset = [load_dag_dataset(dir_path, i) for i in range(max_i + 1)]
	for i, data in enumerate(dataset):
		print(f'Processing data {i:>3}')
		adj_true = data[1]

		adj_pred = train_on_data(data[0], adj_true)

		np.save(
			os.path.join(
				results_sub_folder,
				f'{str(i).zfill(2)}_adj_pred_{FILE_LABEL}'
			),
			adj_pred
		)


def run_synthetic_data():
	for filename in os.listdir(DATA_FOLDER):
		filepath = os.path.join(DATA_FOLDER, filename)
		if os.path.isdir(filepath):
			print(f'\n Processing data {filename}')
			process_dag_data(filepath, filename)


def run_syntren_data():
	process_dag_data(
		SYNTREN_DATA_FOLDER,
		None,
		results_sub_folder=SYNTREN_RESULTS_FOLDER
	)


def run_sachs_data():
	X, adj_true = get_sachs_data(REAL_DATA)[0]

	np.save(
		os.path.join(
			SACHS_RESULTS_FOLDER,
			'Sachs_adj_true.npy'
		),
		adj_true
	)

	adj_pred = train_on_data(X, adj_true, intermediate=True)

	np.save(
		os.path.join(
			SACHS_RESULTS_FOLDER,
			f'Sachs_adj_pred_{FILE_LABEL}.npy'
		),
		adj_pred
	)


if __name__ == '__main__':
	if REAL_DATA is None:
		run_synthetic_data()
	elif REAL_DATA.startswith('Sachs'):
		run_sachs_data()
	elif REAL_DATA == 'syntren':
		run_syntren_data()
