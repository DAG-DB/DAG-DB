""" NoTearsZRegularizer as used in thesis, plus BaseZRegularizer and also
experimental regularizers which were discarded
"""

from abc import ABC, abstractmethod

import torch


class BaseZRegularizer(ABC):
	def __init__(self, d, rho, mu=0., device=None): 
		super().__init__()
		self.d = d
		self.rho = rho
		self.mu = mu
		self.device = device

	@abstractmethod
	def dag_regularizer(self, z_adj):
		raise NotImplementedError

	@abstractmethod
	def sparsity_regularizer(self, z_adj):
		raise NotImplementedError

	def __call__(self, z_adj):
		"""
		Multiply by appropriate constant and add
		:param z_adj:
		:return:
		"""
		res_dag = self.rho * self.dag_regularizer(z_adj)
		if self.mu != 0:
			res_sparsity = self.mu * \
				self.sparsity_regularizer(z_adj)
		else:
			res_sparsity = 0
		return res_dag + res_sparsity


class NoTearsZRegularizer(BaseZRegularizer):
	def dag_regularizer(self, z_adj):
		return 0.5 * torch.square(
			torch.einsum('sii -> s', torch.linalg.matrix_exp(z_adj)) - self.d)

	def sparsity_regularizer(self, z_adj):
		return torch.sum(z_adj, dim=(-2, -1))


class GolemZRegularizer(BaseZRegularizer):  # Give inf sometimes
	""" Really part of the GOLEM loss function rather than a separate
	regularizer so can't use in this way
	"""
	def dag_regularizer(self, z_adj):
		res = torch.linalg.slogdet(torch.eye(self.d, device=self.device)
									- z_adj).logabsdet
		return res


class McKayZRegularizer(BaseZRegularizer):
	""" Experimental regularizer inspire by McKay et al., Acyclic Digraphs and
	Eigenvalues of (0,1)-Matrices}, 2003.  Didn't prove useful
	"""
	def dag_regularizer(self, z_adj):
		res = torch.linalg.slogdet(
			z_adj + torch.eye(self.d, device=self.device)).logabsdet
		return res

	def sparsity_regularizer(self, z_adj):
		return torch.sum(z_adj, dim=(-2, -1))


if __name__ == '__main__':
	reg = McKayZRegularizer(5, 1, device=torch.device('cuda'))
	dist = torch.distributions.bernoulli.Bernoulli(probs=0.4 * torch.ones((
		1, 5, 5), device=torch.device('cuda')))
	# z_adj = torch.distributions.Bernoulli((1, 5, 5), device=torch.device('cuda'))
	z_adj = dist.sample()
	z_adj_n = z_adj.detach().cpu().numpy()
	res = reg(z_adj)
	print(res)
