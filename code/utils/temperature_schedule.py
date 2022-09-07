""" Experimental.  Not used as enough hyperparameters already, and the
probability distribution in any case concentrates around the MAP
"""


import math
from abc import abstractmethod, ABC


class Temperature:
	def __init__(self, tau):
		self.tau = tau

	# The following dunders minimise changes to TargetDistribution and avoid
	# need for changes to imle
	def __mul__(self, other):
		return self.tau * other

	def __rmul__(self, other):
		return other * self.tau

	def __gt__(self, other):
		return self.tau > other


class BaseTemperatureSchedule(ABC):
	def __init__(self, temperature):
		self.temperature = temperature

	@staticmethod
	def negative_end_epoch(end_epoch, n_epochs):
		return end_epoch if end_epoch > 0 else n_epochs + 1 + end_epoch

	@abstractmethod
	def step(self, epoch):
		raise NotImplementedError


class TemperatureGeometricSchedule(BaseTemperatureSchedule):
	def __init__(
			self, temperature, final_multiple, start_epoch, end_epoch,
			n_epochs):
		super().__init__(temperature)
		self.final_multiple = final_multiple
		self.start_epoch = start_epoch
		self.end_epoch = BaseTemperatureSchedule.negative_end_epoch(
			end_epoch, n_epochs)
		self.n_epochs = n_epochs

		assert end_epoch > start_epoch
		self.multiple = math.pow(
			final_multiple, (1 / (end_epoch - start_epoch)))

	def step(self, epoch):
		if (self.start_epoch <= epoch) and (epoch < self.end_epoch):
			self.temperature *= self.multiple
