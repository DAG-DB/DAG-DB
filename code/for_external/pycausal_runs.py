""" Place in the src directory of https://github.com/bd2kccd/py-causal ,
development branch, commit 990dd78114d1cf61c637b69d1b009c93d0d8021e  Follow
README instructions for installation of pycausal (using Python 3.6).

Also place the DAG-DB file code/data_management/get_real_data.py in the
causal-dag src directory.

Then place synthetic and/or real data in DATA_FOLDER (see file path below).
"""


import os

import numpy as np
import pandas as pd


ALGORITHM = 'PC'
assert ALGORITHM in ['FGES', 'PC']

# As recommended by GOLEM paper:
SETTINGS_PARAMETERS ={
	'FGES': {'algoId':'fges', 'scoreId':'cg-bic-score'},
	'PC': {'algoId': 'pc-all', 'testId': 'fisher-z-test'}
}
settings_parameters = SETTINGS_PARAMETERS[ALGORITHM]
# Not entirely clear what settings GOLEM used:
SETTINGS_RUN = {
	# See getAlgorithmParameters output in example/py-causal - FGES Continuous
	 # in Action.ipynb
	'FGES': {'algoId': 'fges', 'maxDegree': -1},
	# See https://github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/java/edu/cmu/tetrad/search/PcAll.java
	 # which suggests defaults to Conservative PC as in GOLEM, so  I chose
	 # values as indicated below
	'PC': {'algoId': 'pc-all',
		   'testId': 'fisher-z-test',  # ad in GOLEM
		   'stableFAS': True,  # for PC-Stable
		   'colliderDiscoveryRule': 1,  # for PC rather than CPC
		   'conflictRule': 2  # to ensure CPDAG result
		   }
}
settings_run = SETTINGS_RUN[ALGORITHM]
FILE_LABEL = ALGORITHM


DAG_FOLDER = os.path.join('..', 'i_mle_dag')
DATA_FOLDER = os.path.join(DAG_FOLDER, 'data')
RESULTS_SUPER_FOLDER = os.path.join(DAG_FOLDER, 'results')
RESULTS_FOLDER = os.path.join(RESULTS_SUPER_FOLDER, FILE_LABEL)
for folder in [RESULTS_SUPER_FOLDER, RESULTS_FOLDER]:
	if not os.path.exists(folder):
		os.mkdir(folder)


def load_dag_dataset(folder_name, i):
	file_stem = os.path.join(folder_name, f'{str(i).zfill(2)}_')
	X = pd.DataFrame(np.load(file_stem + 'data.npy'))
	return X


def process_dag_data(tetrad, dir_path, dir_name):
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
		d = len(data.columns)
		tetrad.run(dfs=data, verbose=False, **settings_run)
		edges = tetrad.getEdges()
		adj_pred = np.zeros((d, d))
		for edge in edges:
			edge_split = edge.split()
			if edge_split[1] in ['---', '<->']:
				edge_value = - 1
			elif edge_split[1] == '-->':
				edge_value = 1
			else:
				raise NotImplementedError
			adj_pred[int(edge_split[0]), int(edge_split[2])] = edge_value

		np.save(
			os.path.join(
				results_sub_folder,
				f'{str(i).zfill(2)}_adj_pred_{FILE_LABEL}'
			),
			adj_pred
		)


def main():
	from pycausal.pycausal import pycausal as pc
	from pycausal import search as s
	pc = pc()
	pc.start_vm()

	tetrad = s.tetradrunner()
	tetrad.getAlgorithmParameters(**settings_parameters)

	for filename in os.listdir(DATA_FOLDER):
		filepath = os.path.join(DATA_FOLDER, filename)
		if os.path.isdir(filepath):
			print(f'\n Processing data {filename}')
			process_dag_data(tetrad, filepath, filename)

	pc.stop_vm()


if __name__ == '__main__':
	main()
