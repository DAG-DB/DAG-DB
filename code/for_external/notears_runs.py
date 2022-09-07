""" Place in the 'notears' sub-directory of
https://github.com/xunzheng/notears ,
master branch, commit ba61337bd0e5410c04cc708be57affc191a8c424

Follow README instructions for installation of NOTEARS with Python 3.6.

Also place the DAG-DB file code/data_management/get_real_data.py in the
same 'notears' sub-directory.

Then place synthetic data in DATA_FOLDER (see file path below) and create a
data directory in notears top level and put the Sachs data directory in it
"""


import os

import numpy as np

from get_real_data import get_real_data
from linear import notears_linear


ALGORITHM = 'NOTEARS-L1'
REAL_DATA = 'Sachs_1'  # = None for synthetic data
FILE_LABEL = ALGORITHM


DAG_FOLDER = os.path.join('..', 'i_mle_dag')
DATA_FOLDER = os.path.join(DAG_FOLDER, 'data')
RESULTS_SUPER_FOLDER = os.path.join(DAG_FOLDER, 'results')
SACHS_RESULTS_FOLDER = os.path.join(RESULTS_SUPER_FOLDER, 'sachs')
RESULTS_FOLDER = os.path.join(RESULTS_SUPER_FOLDER, FILE_LABEL)
for folder in [
	RESULTS_SUPER_FOLDER, SACHS_RESULTS_FOLDER, RESULTS_FOLDER]:
	if not os.path.exists(folder):
		os.mkdir(folder)


def load_dag_dataset(folder_name, i):
	file_stem = os.path.join(folder_name, f'{str(i).zfill(2)}_')
	X = np.load(file_stem + 'data.npy')
	return X


def process_dag_data(dir_path, dir_name):
	results_sub_folder = os.path.join(RESULTS_FOLDER, dir_name)
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
		adj_pred = notears_linear(data, lambda1=0.1, loss_type='l2')

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


def run_real_data():
	X, adj_true = get_real_data(REAL_DATA)[0]

	np.save(
		os.path.join(
			SACHS_RESULTS_FOLDER,
			'Sachs_adj_true.npy'
		),
		adj_true
	)

	adj_pred = notears_linear(X, lambda1=0.1, loss_type='l2')

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
	else:
		run_real_data()
