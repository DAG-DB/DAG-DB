""" Get the Sachs data and true DAG
"""

import os

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def get_consensus_Sachs_graph(draw=False):
	"""
    Get the consensus graph as in fig. 3A of Sachs 2005: note there are
    actually 18 edges, not 17 as the GOLEM paper says
    :return:
	"""
	edges = {  # parents are keys, list of, children are values
        0: [1],
        1: [5],
        2: [3],
        3: [],
        4: [2, 3],
        5: [6],
        6: [],
        7: [0, 1, 5, 6, 9, 10],
        8: [0, 1, 7, 9, 10],
        9: [],
        10: []
	}
	adj_true = np.zeros((11, 11))
	for parent, children in edges.items():
		for child in children:
			adj_true[parent, child] = 1
	if draw:
		G = nx.DiGraph(adj_true)
		nx.draw(G, with_labels=True, font_weight='bold')
		plt.show()
	return adj_true


def get_sachs(file_nos_to_load=None):
	"""
    Loads real Sachs observational data

    Defaults to loading all the Sachs datasets, but can load  only those in
    file_nos_to_load

    :param file_nos_to_load:  None or tuple of int, which files to load by
    their no.  If none, loads all
    :return: tuple of:
     - numpy (*, 11) float, where * = 853 for
     default.  This is the data in the files corresponding to
     file_nos_to_load vstacked into a single numpy array.
     = numpy array of (11, 11) {0, 1}, the true Sachs adjacency matrix
    """

	file_nos_to_load = tuple(list(range(1, 14 + 1))) \
		if file_nos_to_load is None else file_nos_to_load
	data_dir = os.path.join(
		'..', 'data', 'Causal-Protein-Signaling-Sachs+-2005')

	# https://www.geeksforgeeks.org/python-loop-through-files-of-certain-extensions/

    # giving file extension
	ext = ('.xls')

	def _get_file_no(file):
		return int(file[: file.index('.')])

	# iterating over all files
	file_names = dict()
	for file_name in os.listdir(data_dir):
		if file_name.endswith(ext):
			file_names[_get_file_no(file_name)] = file_name
		else:
			continue

	real_data = pd.DataFrame()
	for file_no in file_nos_to_load:
		file_df = pd.read_excel(os.path.join(data_dir, file_names[file_no]))
		file_df = file_df.rename(columns={'pip2': 'PIP2', 'pip3': 'PIP3'})
		print(file_names[file_no], len(file_df))
		if len(real_data) == 0:
			real_data = file_df
		else:
			real_data = pd.concat([real_data, file_df])

	print(f'Totals {len(real_data)} data points.')
	X = real_data.to_numpy()
	adj_true = get_consensus_Sachs_graph()

	return X, adj_true




def get_sachs_data(name):
	"""
	Helper function for get_sachs

	:param name: 'Sachs' (all Sachs data) or 'Sachs_1' (just the purely
	 observational data)
	:return: tuple from return of get_sachs, but wrapped as one-element list
	 for consistency with how datasets are presented in code/main_dag.py
	"""
	if name == 'Sachs':
		X, adj_true = get_sachs()
	elif name == 'Sachs_1':  # The purely observational data used in thesis
		X, adj_true = get_sachs(file_nos_to_load=(1,))
	else:
		raise NotImplementedError

	return [(X, adj_true)]


if __name__ == '__main__':
	get_sachs_data('Sachs')
