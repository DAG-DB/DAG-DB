""" Base for MAP Solvers in code/learn_infer
"""


from abc import ABC, abstractmethod
from contextlib import contextmanager

import numpy as np


class BaseSolver(ABC):
	def __init__(self, d, s, max_size=None, threshold=None, mode=None):
		self.d = d
		self.s = s
		self.original_s = s
		self.mode = mode.mode
		self.d_range = np.arange(d, dtype=np.int64)
		self.max_size = None if ((max_size is None) or (
				max_size >= d * (d - 1) / 2)) else max_size
		self.threshold = threshold
		self.calculated_threshold = None
		self.test_only = None

	@property
	def s(self):
		return self._s

	@s.setter
	def s(self, value):
		self._s = value
		self.s_range = np.arange(value)
		self.s_indices = np.arange(value).repeat(self.d)
		self.row_indices = np.tile(np.arange(self.d, dtype=np.int64), value)

	def s_reset(self):
		self.s = self.original_s

	@contextmanager
	def inference_solver(self):
		try:
			self.s = 1
			yield None
		finally:
			self.s_reset()

	def get_threshold(self, theta):
		"""
		Get threshold, taking account of self.max_size, self.threshold
		:param theta: (s, *) numpy float32 array
		:return:  (s, 1) numpy float32 array or float
		"""
		if (self.max_size is not None) and (self.max_size < theta.shape[1]):
			# threshold is the (self.max_size + 1)th largest in dim -1,
			# and of shape (s, 1)
			s_range = list(range(theta.shape[0]))
			threshold = theta[
							s_range, np.argsort(theta, axis=-1)[
										  :, - self.max_size - 1]
						][:, None]
			threshold = np.maximum(self.threshold, threshold)
		else:
			threshold = self.threshold
		self.calculated_threshold = threshold
		return threshold

	@abstractmethod
	def __call__(self, theta):
		raise NotImplementedError


class BaseSolverScheduler(ABC):
	def __init__(self, solver, n_epochs):
		self.solver = solver
		self.n_epochs = n_epochs
		self.epoch = 1
		self._decide_for_epoch()

	def step(self):
		self.epoch += 1
		self._decide_for_epoch()

	@abstractmethod
	def _decide_for_epoch(self):
		"""
		Must set self.solver.test_only True or False according to
		schedule's condition on self.epoch
		"""
		raise NotImplementedError
