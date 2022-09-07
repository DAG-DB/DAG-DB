""" The MiniZinc Solver, with thanks to Pasquale Minervini for the key
MiniZincMaximumDAG solver.

Used experimentally, including to confirm the approximate max DAG solver, AFAS.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

os.environ["PATH"] += os.pathsep + \
					  "/home/andrew/MiniZincIDE-2.6.4-bundle-linux-x86_64/bin"

import networkx as nx
from itertools import chain, combinations

import numpy as np
from minizinc import Instance, Model, Solver
import torch

from utils.base_solver import BaseSolver

import logging


logger = logging.getLogger(os.path.basename(sys.argv[0]))


class MiniZincMaximumDAG:
	def __init__(self, d, s):
		self.d = d
		self.s = s
		self.model = Model()
		self.model.add_string(
			'''
			include "dag.mzn";

			predicate weighted_dag(array[$$E] of $$N: from, array[$$E] of $$N: to, array[int] of float: w,
								   array[int] of var bool: ns, array[int] of var bool: es, var float: K) =
			   assert(index_set(from) = index_set(to),"dreachable: index set of from and to must be identical") /\\
			   assert(index_set(from) = index_set(es),"dreachable: index set of from and es must be identical") /\\
			   assert(dom_array(from) subset index_set(ns),"dreachable: nodes in from must be in index set of ns") /\\
			   assert(dom_array(to) subset index_set(ns),"dreachable: nodes in to must be in index set of ns") /\\
			   dag(from, to, ns, es) /\\ K = sum(e in 1..length(es)) (es[e] * w[e]); 

			array [int] of int: from;
			array [int] of int: to;
			array [int] of float: w;

			array [int] of int: nodes;

			% Find the maximum sub-DAG of the graph
			var float: total_weight;

			array [index_set(from)] of var bool: es;
			array [index_set(nodes)] of var bool: ns;

			constraint ns == [true | n in ns] /\\ weighted_dag(
			  from,         % Edge from node
			  to,           % Edge to node
			  w,            % Weight of edge
			  ns,           % Whether node is in dag
			  es,           % Whether edge is in dag
			  total_weight  % Total weight of dag
			);

			solve maximize total_weight;
			''')

		# solver = Solver.lookup()
		self.solver = Solver.lookup("mip")
		# solver.stdFlags = ["-p", "8"]

	def max_dag(self, weighted_digraph):
		weighted_digraph_dict = {(i + 1, j + 1): weighted_digraph[i, j]
							for i in range(self.d) for j in range(self.d)
								 if (j != i) and (weighted_digraph[i, j] > 0)}

		instance = Instance(self.solver, self.model)

		edge_lst = [edge for edge in weighted_digraph_dict.keys()]

		instance["from"] = [edge[0] for edge in edge_lst]
		instance["to"] = [edge[1] for edge in edge_lst]
		instance["w"] = [weighted_digraph_dict[edge] for edge in edge_lst]
		instance["nodes"] = list(range(1, self.d + 1))

		result = instance.solve(
			intermediate_solutions=False, all_solutions=False)
		es = result["es"]

		edges = [edge for i, edge in enumerate(edge_lst) if es[i] is True]
		max_dag = np.zeros((self.d, self.d), dtype=np.int64)
		for edge in edges:
			max_dag[edge[0] - 1, edge[1] - 1] = 1
		return max_dag

	# def __call__(self, theta):
	# 	"""
	#
    #     :param theta: (s, d, d) float32 ndarray
    #     :return: (s, d, d) int64 ndarray, z
	# 	"""
	# 	return np.concatenate(
	# 		[self.max_dag(weighted_digraph) for weighted_digraph in theta])


class MiniZincSolverFinal(BaseSolver):
	def __init__(self, d, s, max_size=None, threshold=None, mode=None):
		super().__init__(d, s, max_size, threshold, mode)
		self.minizinc = MiniZincMaximumDAG(d, s)

	def test_get_threshold(self, theta):
		"""
		Get threshold, taking account of self.max_size, self.threshold
		:param theta: (d ** 2,) numpy float32 array
		:return: float
		"""
		if self.max_size is not None:
			# threshold is the (self.max_size + 1)th largest in dim -1,
			# and of shape (s, 1)
			threshold = theta[np.argsort(theta, axis=-1)[
										  - self.max_size - 1]
						]
			threshold = np.maximum(self.threshold, threshold)
		else:
			threshold = self.threshold
		self.calculated_threshold = threshold
		return threshold

	def test(self, z, theta):
		z = z.squeeze()
		if isinstance(z, torch.Tensor):
			z = z.detach().cpu().numpy()
		if len(z.shape) == 1:
			z = z.reshape(self.d, self.d)
		if isinstance(theta, torch.Tensor):
			theta = theta.detach().cpu().numpy()
		threshold = self.test_get_threshold(theta.flatten())
		theta = theta * (theta > threshold)
		theta = theta.reshape(self.d, self.d)
		for i in range(self.d):
			theta[i, i] = 0

		z_dag = self.minizinc.max_dag(z * theta)

		return z_dag

	def __call__(self, theta):
		raise NotImplementedError

