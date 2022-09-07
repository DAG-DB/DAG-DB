""" For convenience wrap h and f optimizers into single optimizer-like class
"""


from lib.ml_utilities import get_optimizer
from lib.ml_utilities import h  # Used dynamically


class hfOptimizer:
	def __init__(self, oa_net, h_opt, f_opt):
		self.h_opt = h_opt
		self.f_opt = f_opt

	def zero_grad(self):
		self.h_opt.zero_grad()
		self.f_opt.zero_grad()

	def step(self):
		self.h_opt.step()
		self.f_opt.step()
