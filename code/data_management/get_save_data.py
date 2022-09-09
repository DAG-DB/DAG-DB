""" Utility code to support creation of synthetic data, and use of synthetic
and real data
"""


import os

import numpy as np

import lib.ml_utilities as mlu
from lib.ml_utilities import c, h
from data_management.synthetic_dataset import SyntheticDataset
from data_management.get_real_data import get_sachs_data


def get_folder(data_category, folder=None):
	metadata = f'{str(h.D).zfill(3)}_{h.GRAPH_TYPE}_{h.SEM_NOISE}_'
	folder = metadata + mlu.LOG_FILESTEM if folder is None else folder
	folder = os.path.join(c.DATA_FOLDER, data_category, folder)
	return folder


def save_dag(data_category, X, B, i):
	folder = get_folder(data_category)
	if not os.path.exists(folder):
		os.mkdir(folder)
	file_stem = os.path.join(folder, f'{str(i).zfill(2)}_')
	np.save(file_stem + 'data.npy', X)
	np.save(file_stem + 'adj.npy', B)


def load_dag(folder_name, i):
	file_stem = os.path.join(folder_name, f'{str(i).zfill(2)}_')
	X = np.load(file_stem + 'data.npy')
	B = np.load(file_stem + 'adj.npy')
	return X, B


def generate_dag_data(data_category, i, seed):
	assert data_category in ['train', 'test', 'v_methods_test']
	match h.GRAPH_TYPE[:2]:
		case 'ER':
			graph_type = 'ER'
			graph_parameter = int(h.GRAPH_TYPE[2])
		case 'SF':
			graph_type = 'SF'
			graph_parameter = int(h.GRAPH_TYPE[2])
		case _:
			raise NotImplementedError
	data = SyntheticDataset(
		h.N, h.D, graph_type, 2 * graph_parameter, h.SEM_NOISE, 1., seed=seed)
	save_dag(data_category, data.X, data.B, i)
	return data.X, data.B


def load_dag_data(data_category, dags, n_real_runs=1):
	if data_category == 'real':
		return get_sachs_data(dags, n_real_runs)
	assert isinstance(dags, str)
	u0 = dags.find('_')
	u1 = dags[u0 + 1:].find('_') + u0 + 1
	if h.D is None:
		h.D = int(dags[:u0])
	if h.GRAPH_TYPE is None:
		h.GRAPH_TYPE = dags[u0 + 1: u1]
	if not (data_category  in ['big', 'syntren']):
		assert h.D == int(dags[:u0])
		assert h.GRAPH_TYPE == dags[u0 + 1: u1]
		assert dags[u1 + 1:].startswith(h.SEM_NOISE)

	folder_name = os.path.join(
		c.DATA_FOLDER, data_category, dags)
	max_i = -1
	for filename in os.listdir(folder_name):
		if not filename.endswith('.npy') or (max_i >= int(filename[:2])):
			continue
		max_i = int(filename[:2])

	data = [load_dag(folder_name, i) for i in range(max_i + 1)]
	return data

