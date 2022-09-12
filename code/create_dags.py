""" The main program for generating synthetic DAGs and datasets.

To use construct a config file, like code/named_configs/create_dags.py
which just needs those hyperparameters, in particular:
- 'DATA_CATEGORY': the sub-directory of ./data where a sub-directory
	will be placed containing the DAGs and data.
- 'GRAPH_TYPE', 'D', 'SEM_NOISE', 'N' and 'VAL' as appropriate,
- 'DAGS': a list of random seeds, one for each DAG/dataset to be created.
	A list of a required length can be generated
	using code/lib/random_seed_generator.py  For ER graphs, these seeds are
	used.  For SF graphs, seeds cannot be used and just corresponds to the
	number of DAG/datasets required.
"""


import shutil

import lib.ml_utilities as mlu
from lib.ml_utilities import h
from data_management.get_save_data import generate_dag_data, get_folder, load_dag_data


def create_save():
	assert isinstance(h.DAGS, list)
	for i, seed in enumerate(h.DAGS):
		assert isinstance(seed, int)
		generate_dag_data(h.DATA_CATEGORY, i, seed)

	mlu.logging.shutdown()
	folder = get_folder(h.DATA_CATEGORY)
	shutil.move(mlu.LOG_FILEPATH, folder)


if __name__ == '__main__':
	create_save()

