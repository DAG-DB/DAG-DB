""" Formerly used to record info about the latent variable Z at per-batch level
"""


import lzma
import json
import math
import os
from collections import namedtuple
from functools import wraps

import networkx as nx
import numpy as np

from lib.ml_utilities import c, get_latents_save_name
from utils.metrics import count_accuracy


class LatentExhibit:
	def __init__(self, d, name=None):
		self.d = d
		self.width = {
			'lt': (self.d * (self.d - 1)) // 2,
			'p': self.d ** 2
		}

		if name is None:
			self.z = {
				'lt': None,
				'p': None
			}
		else:
			self._load(name)

		self.initial = {
			'lt': True,
			'p': True
		}
		self.adj_matrix_z = None
		self.graph = None

	def _update(self, z, key):
		if self.initial[key]:
			self.initial[key] = False
			return z
		return np.append(self.z[key], z, axis=0)

	def update(self, z):
		# Ensure that will not use old values:
		self.adj_matrix_z = self.graph = None
		z = z.detach().cpu().numpy().astype(np.int64)
		z_lt = z[..., :self.width['lt']]
		z_p = z[..., self.width['lt']:]
		self.z['lt'] = self._update(z_lt, 'lt')
		self.z['p'] = self._update(z_p, 'p')

	# noinspection PyTypeChecker
	def _pack(self):
		return {key: np.packbits(value, axis=-1)
						 for key, value in self.z.items()}

	def save(self, tag):
		path = get_latents_save_name(tag)
		# # for testing:
		# path = os.path.join('..', '..', 'latents', tag + '_latents.json')
		z_packed = self._pack()
		to_save = {
			'd': self.d,
			**{key: value.tolist() for key, value in z_packed.items()}
			   }
		with lzma.open(path, 'wt') as fp:
			# noinspection PyUnresolvedReferences
			json.dump(to_save, fp)

	# noinspection PyUnresolvedReferences
	def _load(self, name):
		path = os.path.join(c.LATENTS_FOLDER, name + '.lzma')
		# # for testing:
		# path = os.path.join('..', '..', 'latents', name + '_latents.json')
		with lzma.open(path, 'rt') as fp:
			loaded = json.load(fp)
		assert  self.d == loaded['d']
		del loaded['d']
		self.z = {key: np.unpackbits(np.array(value, dtype=np.uint8), axis=-1,
									 count=self.width[key])
						 for key, value in loaded.items()}

	def _tuple_pack(self, z):
		z = np.packbits(z, axis=-1)
		dtypes = [(str(i), np.uint8) for i in range(z.shape[-1])]
		return z.view(dtype=np.dtype(dtypes)  # thanks to https://stackoverflow.com/questions/43426039/convert-last-dimension-of-ndarray-to-tuple-recarray
				  ).squeeze(-1)

	def reshape(self, z_shape_stem):
		for key in self.z:
			self.z[key] = self.z[key].reshape(
				z_shape_stem + (self.width[key],))

	Labels = namedtuple('Labels', 'labelled label2hash idx2hash')

	def label(self, z):
		z = self._tuple_pack(z)
		z_shape = z.shape
		z = z.flatten()
		idx2hash = np.array([hash(tuple(elt)) for elt in z])
		label2hash = np.unique(idx2hash)
		labelled = np.digitize(idx2hash, label2hash, right=True)
		return self.Labels(
			labelled.reshape(z_shape), label2hash, idx2hash.reshape(z_shape))

	"""rsion of self.label below is 
	"""
	def _slow_label(self, z, labelling):
		if z in labelling:
			return labelling.index(z)
		labelling.append(z)
		return len(labelling) - 1

	def slow_label(self, z):
		""" Much slower than self.label() (Ok for d=30 on validation set) but
		gets the labels in ascending order
		"""
		z = self._tuple_pack(z)
		z_shape = z.shape
		z = z.flatten()
		labelling = list()
		labelled = np.array([self._slow_label(elt, labelling) for elt in list(z)]
							).reshape(z_shape)
		labelling = np.array(labelling)
		return self.Labels(labelled, labelling, None)

	def size(self):
		# noinspection PyUnresolvedReferences
		self.to_adjacency_matrix()
		return self.adj_matrix_z.sum(axis=(-2, -1))

	def to_adjacency_matrix(self):
		# noinspection PyUnresolvedReferences
		if self.adj_matrix_z is None:
			matrix_z_lt = np.zeros(
				(self.z['lt'].shape[0], self.z['lt'].shape[1], self.d, self.d),
				dtype=np.int64
			)
			tril_indices = np.tril_indices(self.d, k=-1)
			matrix_z_lt[:, :, tril_indices[0], tril_indices[1]] = self.z['lt']
			matrix_z_p = self.z['p'].reshape(
				self.z['p'].shape[:2] + (self.d, self.d))
			self.adj_matrix_z = matrix_z_p.transpose(
				0, 1, 3, 2) @ matrix_z_lt @ matrix_z_p

	DegreeSequence = namedtuple('DegreeSequence', 'd_in d_out')

	def degree_sequence(self):
		self.to_adjacency_matrix()
		degree_in = self.adj_matrix_z.sum(axis=-2)
		degree_out = self.adj_matrix_z.sum(axis=-1)
		return self.DegreeSequence(degree_in, degree_out)

	def mean_degree(self):
		return 2 * self.size() / self.d

	def n_roots(self):
		return (self.degree_sequence().d_in == 0).sum(axis=-1)

	def n_leaves(self):
		return (self.degree_sequence().d_out == 0).sum(axis=-1)

	def to_graph(self):
		"""
		Note that self.graph is flattened as (n * s,) list
		"""
		self.to_adjacency_matrix()
		if self.graph is None:
			adj_matrix_z = self.adj_matrix_z.reshape(-1, self.d, self.d)
			self.graph = [nx.DiGraph(matrix) for matrix in adj_matrix_z]

	def get_graph_property(func):
		# noinspection PyTypeChecker
		@wraps(func)
		def wrapper(self, *args, **kwargs):
			self.to_graph()
			return np.array([func(graph) for graph in
						 self.graph], dtype='object').reshape(
				self.adj_matrix_z.shape[:2])
		return wrapper

	# noinspection PyArgumentList
	@get_graph_property
	def n_cc(graph):
		return nx.number_connected_components(nx.to_undirected(graph))

	# noinspection PyArgumentList
	@get_graph_property
	def topological_generations(graph):
		return tuple(nx.topological_generations(graph))

	# noinspection PyArgumentList
	@get_graph_property
	def topological_generation_sizes(graph):
		return tuple(len(gen) for gen in nx.topological_generations(graph))

	def compare(self, z_lt_base, z_lt):
		z_lt_shape = z_lt.shape
		metrics = [count_accuracy(z_lt_base, matrix)
				   for matrix in z_lt.reshape(-1, self.d, self.d)]
		return {key: np.array([elt[key] for elt in metrics]).reshape(
			z_lt_shape[:2])
				for key in ['shd', 'shd_c']}

	def er4_log_prob(self):
		er = 4
		p = er / (self.d - 1)
		size_on = self.size()
		size_off = self.width['lt'] - size_on
		return size_on * math.log(p) + size_off * math.log(1 - p)


class GraphExhibit(LatentExhibit):
	def __init__(self, d, adjacency_true=None, name=None):
		super().__init__(d, name)
		if (name is None) and (adjacency_true is not None):
			self.adj_matrix_z = adjacency_true[None, None, :, :]
			self.z = None

	def update(self, z):
		pass

	def to_adjacency_matrix(self):
		pass

	def get_graph_property(func):
		# noinspection PyTypeChecker
		@wraps(func)
		def wrapper(self, *args, **kwargs):
			self.to_graph()
			return [func(graph) for graph in self.graph][0]
		return wrapper

	# noinspection PyArgumentList
	@get_graph_property
	def n_cc(graph):
		return nx.number_connected_components(nx.to_undirected(graph))

	# noinspection PyArgumentList
	@get_graph_property
	def topological_generations(graph):
		return tuple(nx.topological_generations(graph))

	# noinspection PyArgumentList
	@get_graph_property
	def topological_generation_sizes(graph):
		return tuple(len(gen) for gen in nx.topological_generations(graph))

if __name__ == '__main__':
	# rng = np.random.default_rng(seed=1882)
	# z_exhibit = LatentExhibit(3)
	# z = rng.integers(2, size=(5, 7, 12))
	# z_exhibit.update(z)
	# z = rng.integers(2, size=(5, 7, 12))
	# z_exhibit.update(z)
	# # noinspection PyTypeChecker
	# labels = z_exhibit.label(z_exhibit.z['lt'])
	# z_exhibit.save('name')

	# z_exhibit = LatentExhibit(3, name='name')

	pass
