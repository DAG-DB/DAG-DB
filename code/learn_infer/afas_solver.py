""" The GFAS approximate maximum DAG solver. Adapted from
https://github.com/stamps/FAS/blob/master/BergerShorWeightedFAS.java
for Java to Python, and for arbitrary *positive* edge weights
"""

from contextlib import contextmanager

import networkx as nx
import numpy as np
import torch

from utils.base_solver import BaseSolver, BaseSolverScheduler


class ArrayWeightedFAS:
	""" The core solver adapted from
	https://github.com/stamps/FAS/blob/master/BergerShorWeightedFAS.java
	for Java to Python, and for arbitrary *positive* edge weights
	"""
	def __init__(self, G, d):
		# G.remove_edges_from([
		# 	edge for edge in G.edges() if G[edge[0]][edge[1]]['weight'] <= 0])
		self.orig_G_positive = G
		self.G = G.copy()
		self.d = d
		self.deltas = np.zeros(d)  # will give delta for each node when
		# populated
		self.calculate_deltas()
		self.sinks = None
		self.sources = None
		self.update_sinks_sources()
		self.max_delta = - np.inf
		self.max_delta_node = None
		self.update_max_delta(- np.inf)
		self.seq = None

	def get_weight(self, tail, head):
		return self.orig_G_positive.get_edge_data(tail, head)['weight']

	def weight_out(self, v):
		return np.sum([
			self.get_weight(v, u)
			for u in self.orig_G_positive.successors(v)])

	def weight_in(self, v):
		return np.sum([
			self.get_weight(u, v)
			for u in self.orig_G_positive.predecessors(v)])

	def calculate_deltas(self):
		for v in self.G.nodes():
			self.deltas[v] = self.weight_out(v) - self.weight_in(v)

	def update_sinks_sources(self):
		self.sinks = set(v for v in self.G.nodes()
						 if self.G.out_degree(v) == 0)
		self.sources = set(v for v in self.G.nodes()
						   if self.G.in_degree(v) == 0)

	def update_max_delta(self, delta):
		if delta == self.max_delta:
			self.max_delta_node = np.argmax(self.deltas)
			self.max_delta = self.deltas[self.max_delta_node]
			pass

	def delete_node(self, u):
		old_delta = self.deltas[u]
		self.deltas[u] = - np.inf
		self.update_max_delta(old_delta)
		self._delete_node(u, True)
		self._delete_node(u, False)
		self.G.remove_node(u)
		self.update_sinks_sources()

	def _delete_node(self, u, out):
		if out:
			u_neighbours = self.G.successors(u)
		else:
			u_neighbours = self.G.predecessors(u)
		for v in u_neighbours:
			if v == u:
				continue
			old_delta = self.deltas[v]
			if out:
				self.deltas[v] += self.get_weight(u, v)
			else:
				self.deltas[v] -= self.get_weight(v, u)
			self.update_max_delta(old_delta)

	def computeseq(self):
		"""
		The core of the algorithm, putting the vertices into a sequence
		"""
		s1 = list()
		s2 = list()

		numdel = 0
		while numdel < self.d:
			while self.sinks != set():
				u = self.sinks.pop()
				# u = min([(u, self.deltas[u]) for u in self.sinks],
				# 		key= lambda elt: elt[1])[0]
				self.delete_node(u)
				numdel += 1
				s2 = [u] + s2

			while self.sources != set():
				u = self.sources.pop()
				# u = max([(u, self.deltas[u]) for u in self.sources],
				# 		key= lambda elt: elt[1])[0]
				self.delete_node(u)
				numdel += 1
				s1 = s1 + [u]

			if numdel < self.d:
				u = self.max_delta_node
				self.delete_node(u)
				# self.update_max_delta(self.max_delta)
				numdel += 1
				s1 = s1 + [u]

		s1.extend(s2)
		self.seq = s1

	def compute_FAS(self):
		if self.seq is None:
			self.computeseq()

		v_array = - np.ones(self.d, dtype=np.int64)
		i = 0
		for u in self.seq:
			v_array[u] = i
			i += 1

		fvs = np.zeros(self.d, dtype=np.int64)
		fas_wt = 0
		fas = list()
		own = 0

		for v in self.orig_G_positive.nodes():
			children = list(self.orig_G_positive.successors(v))

			for w in children:
				if v == w:  # self-loop ignore
					own += 1
					continue

				if v_array[v] > v_array[w]:
					fvs[v] = 1
					fas.append((v, w))
					fas_wt += self.get_weight(v, w)

		# print(f'fvs size is {fvs.sum()}')
		# print(f'fas is {fas}')
		# print(f'fas weight is {fas_wt}')
		return fas

	def compute_max_DAG(self):
		fas = self.compute_FAS()
		dag = self.orig_G_positive
		dag.remove_edges_from(fas)
		return dag


class AFASSolver(BaseSolver):
	""" The solver using the approx max DAG solve every epoch
	"""
	def __init__(self, d, s, max_size=None, threshold=None, mode=None):
		super().__init__(d, s, max_size, threshold, mode)
		self.test_only = False

	def __call__(self, theta):
		theta = theta.reshape(-1, self.d, self.d)
		for row in theta:
			for i in range(self.d):
				row[i, i] = 0.
		theta = theta.reshape(-1, self.d ** 2)
		threshold = self.get_threshold(theta)
		theta = theta * (theta > threshold)
		theta = theta.reshape(-1, self.d, self.d)

		zs_by_sample = list()
		for theta_sample in theta:
			G = nx.DiGraph()
			G.add_nodes_from(list(range(self.d)))
			G.add_edges_from([
				(i, j, {'weight': theta_sample[i, j]})
				for i in range(self.d) for j in range(self.d)
				if theta_sample[i, j] > 0.
			])
			afas = ArrayWeightedFAS(G, self.d)
			dag = afas.compute_max_DAG()
			z = np.zeros((self.d, self.d), dtype=np.int64)
			for edge in dag.edges():
				z[edge[0], edge[1]] = 1
			zs_by_sample.append(z.flatten())
		return np.vstack((zs_by_sample))


class AFASSolverTestOnly(BaseSolver):
	""" The solver using the approx max DAG solver in evaluation only.  The
	main solver used for the thesis
	"""
	def __init__(self, d, s, max_size=None, threshold=None, mode=None):
		super().__init__(d, s, max_size, threshold, mode)
		self.test_only = True
		self.form_dag_in_infer = False

	def __call__(self, theta):
		theta = theta.reshape(-1, self.d, self.d)
		for row in theta:
			for i in range(self.d):
				row[i, i] = 0.
		theta = theta.reshape(-1, self.d ** 2)
		threshold = self.get_threshold(theta)
		theta = theta * (theta > threshold).astype(np.float32)  # To address
		 # cluster runtime warning

		return (theta > 0).astype(np.int64)

	def test_get_threshold(self, theta):
		"""
		Get threshold, taking account of self.max_size, self.threshold
		:param theta: (d ** 2,) numpy float32 array
		:return: float
		"""
		if (self.max_size is not None) and (self.max_size < theta.shape[0]):
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
		"""
		Given a digraph z with weights theta, use the ArrayWeightedFAS to
		find approx max DAG

		:param z: digraph, ndarray (1, d ** 2), or Tensor (d, d). of ints
		:param theta: weights, ndarray (1, d ** 2) of floats
		:return: approx max DAG
		"""
		z = z.squeeze()
		if isinstance(z, torch.Tensor):
			z = z.detach().cpu().numpy()
		if len(z.shape) == 1:
			z = z.reshape(self.d, self.d)
		threshold = self.test_get_threshold(theta.flatten())
		theta = theta * (theta > threshold)
		theta = theta.reshape(self.d, self.d)

		G = nx.DiGraph()
		G.add_nodes_from(list(range(self.d)))
		G.add_edges_from([
			(i, j, {'weight': theta[i, j]})
			for i in range(self.d) for j in range(self.d)
			if z[i, j] != 0
		])
		afas = ArrayWeightedFAS(G, self.d)
		dag = afas.compute_max_DAG()
		z_dag = np.zeros((self.d, self.d), dtype=np.int64)
		for edge in dag.edges():
			z_dag[edge[0], edge[1]] = 1
		return z_dag

	@contextmanager
	def form_dag_to_infer(self):
		try:
			self.form_dag_in_infer = True
			yield None
		finally:
			self.form_dag_in_infer = False


class AFASSolverTestSchedule(BaseSolver):
	""" Allows an arbitrary schedule of when to find the max DAG.  Did not
	seems helpful, and adds complication
	"""
	def __init__(
			self, d, s, max_size=None, threshold=None, mode=None):
		super().__init__(d, s, max_size, threshold, mode)
		self.test_only = None
		self.solver = {
			False: AFASSolver(d, s, max_size, threshold, mode),
			True: AFASSolverTestOnly(d, s, max_size, threshold, mode)
		}

	def __call__(self, theta):
		return self.solver[self.test_only](theta)

	def test(self, z, theta):
		assert self.test_only
		return self.solver[True].test(z, theta)

	@contextmanager
	def form_dag_to_infer(self):
		try:
			self.form_dag_in_infer = True
			yield None
		finally:
			self.form_dag_in_infer = False


class RegularSolverScheduler(BaseSolverScheduler):
	""" Scheduler which can be used with AFASSolverTestSchedule
	"""
	def __init__(self, solver, n_epochs, *,
				 n_initial=0, n_final=0, period=None):
		self.n_initial = n_initial
		self.n_final = n_final
		self.period = n_epochs + 1 if period is None else period

		self.after_initial = n_initial
		self.start_final = n_epochs - n_final + 1

		super().__init__(solver, n_epochs)

	def _decide_for_epoch(self):
		self.solver.test_only = (
			self.epoch > self.after_initial) and (
				self.epoch < self.start_final) and (
			(self.epoch % self.period) != 0)


if __name__ == '__main__':
	# mode = Mode('max_dag')
	# afas = AFASSolverTestSchedule(30, 10, mode=mode)
	# solver_scheduler = RegularSolverScheduler(
	# 	afas, 10, n_initial=10)
	# for epoch in range(10):
	# 	print(f'{epoch}: {solver_scheduler.solver}')
	# 	solver_scheduler.step()
	pass
