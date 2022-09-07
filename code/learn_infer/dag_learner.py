""" The main learning loop
"""

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils.latent_analysis import LatentExhibit
from discrete_backprop.dag_sfe_reinforce import sfe_set_h_grad
from learn_infer.afas_solver import AFASSolver, AFASSolverTestOnly, \
	AFASSolverTestSchedule
from learn_infer.dag_nets import hNet, hNull, hNetVectorP, hNullVectorP, fNet
from learn_infer.dag_solver import DAGSolver
from learn_infer.d_p_solver import DPSolver
import lib.ml_utilities as mlu
from lib.ml_utilities import c
from utils.metric_handlers import handle_wandb_train, handle_metrics
from utils.split_data import split_train_val


""" Note: the adjacency matrix convention is that the adjacency[i, j] is for 
the edge from i to j.

This is implemented in:
 - learn_infer.dag_nets.fNet.forward via
    lib.channel_linear.ChannelLinear.forward, so that each channel 
    represents a child note (the inputs ultimately coming from parent
	nodes);
 - in data_management.synthetic_dataset.SyntheticDataset:
	- _graph_to_adjmat, which reflects that this is the networkx convention,
		see e.g. which reflect the convention, see e.g.,
		networkx/convert_matrix.py
	- _simulate_single_equation, which then reflects that in creation of the X 
		data. 
"""


class Mode:
	""" Determining various settings which depend on the mode
	"""
	def __init__(self, d, mode):
		self.d = d
		self.mode = mode
		self.theta_width = d ** 2
		match mode:
			case 'lt_p_matrix':
				self.theta_width = d * (d - 1) // 2 + d ** 2
				self.solver = DAGSolver
			case _ if mode.startswith('lt_p_vector'):
				self.theta_width = d * (d - 1) // 2 + d
				if mode == 'lt_p_vector_argsort_in_std_autograd':
					self.solver = DAGSolver
				elif mode == 'lt_p_vector_argsort_in_custom_autograd':
					raise NotImplementedError
			case 'max_dag':
				self.solver = AFASSolver
			case 'max_dag_test_only':
				self.solver = AFASSolverTestOnly
			case 'max_dag_schedule':
				self.solver = AFASSolverTestSchedule
			case 'd_p':
				self.theta_width = d ** 2 + d
				self.solver = DPSolver
			case _:
				raise NotImplementedError


class OverallNet(nn.Module):
	"""	The overall neural net, comprising the hNet which, in the thesis,
	is simply the learnable Theta, and the fNet which, in the thesis,
	is simply a single linear layer
	"""
	def __init__(
			self, d, s, batch_size, device, solver, torch_solver, imle_solver,
			h_linear_layers=None,
			f_linear_layers=None,
			h_null=None,
			f_bias=True,
			mode=None,
	):
		super().__init__()
		self.d = d
		self.device = device
		self.mode = mode
		self.h_null = h_null
		match mode.mode:
			case _ if mode.mode in [
				'lt_p_matrix',
				'max_dag',
				'max_dag_test_only',
				'max_dag_schedule'
			]:
				if h_null is None:
					self.h_net = hNet(
						d, batch_size,
						linear_layers=h_linear_layers,
						theta_width=mode.theta_width
					)
				else:
					self.h_net = hNull(
						d, bound=h_null, theta_width=mode.theta_width)
			case 'lt_p_vector_argsort_in_std_autograd':
				if h_null is None:
					self.h_net = hNetVectorP(
						d, batch_size, device, linear_layers=h_linear_layers)
				else:
					self.h_net = hNullVectorP(d, device, bound=h_null)
			case 'd_p':
				if h_null is None:
					self.h_net = hNet(
						d, batch_size,
						linear_layers=h_linear_layers,
						theta_width=mode.theta_width,
						d_p_flag=True
					)
				else:
					self.h_net = hNull(
						d, bound=h_null, theta_width=mode.theta_width,
						d_p_flag=True)
			case _:
				raise NotImplementedError
		self.f_net = fNet(d, s, device, mode, linear_layers=f_linear_layers,
						  f_bias=f_bias)
		self.solver = solver
		self.torch_solver = torch_solver
		self.imle_solver = imle_solver

	def forward(self, X):
		theta = self.h_net(X)
		z = self.imle_solver(theta)
		return *self.f_net(X, z), z, theta

	def infer(self, X):
		with torch.inference_mode():
			theta = self.h_net(X).detach().cpu().numpy().reshape(1, -1)
			with self.solver.inference_solver():
				z = self.solver(theta)
			if (self.mode.mode == 'max_dag_test_only') and \
					self.solver.form_dag_in_infer:
				z = self.solver.test(z, theta)
			z = torch.tensor(z, dtype=torch.float32, device=self.device)
			X, z_adjacency = self.f_net(X, z)
		return X, z_adjacency, z, theta


def dag_learner(i, X, adjacency_true, overall_net, trial, *, device, val_prop,
				batch_size, n_epochs, optimizer,
				lr_scheduler, z_regularizer, batch_shuffle=None, solver=None,
				wandbs=None, afas_scheduler=None, minizinc_solver=None,
				trial_store=None, noise_temperature=None, sfe_flag=False
				):
	""" The main training loop.

	:param i: The number of the DAG in the whole run from main_dag
	:param X: numpy (N, d) float dataset
	:param adjacency_true: numpy (d, d) int true adjacency matrix (for metrics)
	:param overall_net: OveralLNet instance (see above)
	:param trial: trial if using Optuna, else None
	:param device: torch device: cuda or cpu etc.
	:param val_prop: float: proportion of N used for validation set
	:param batch_size: int
	:param n_epochs: int
	:param optimizer: torch optimizer
	:param lr_scheduler: torch learning rate scheduler
	:param z_regularizer: the regularizer discussed in thesis sec 5.2
	:param batch_shuffle: True or False
	:param solver: MAP solver
	:param wandbs: numpy (n_epochs,) array of dicts to record metrics for wandb
	:param afas_scheduler:  experimental: set to None. Was used to enable
	 the scheduler in code/learn_infer/afas_solver.py
	:param minizinc_solver: experimental: set to None.  Was used to
	 enable the MiniZinc maximum DAG solver in
	  code/learn_infer/minizinc_solver.py
	:param trial_store: utils.metric_handlers.TrialStore instance  used to
	 manage hyperparameters which change per trial.  Needed in case a trial
	 is pruned
	:param noise_temperature: noise temperature for use here with SFE only
	:param sfe_flag: True if using SFE, else False
	:return: last_metrics, a dict of metrics taken at the end of training
	"""
	n, d = X.shape
	if n < batch_size:
		mlu.warning(f'ALERT! {batch_size=} < {n=}, so no learning will happen')
	X = X - X.mean(axis=0, keepdims=True)  # Centre the data
	X_train, X_val = split_train_val(X, device, val_prop)

	X_train_batches = torch.utils.data.DataLoader(
		X_train, batch_size=batch_size, shuffle=batch_shuffle,
		drop_last=True)
	X_val_batches = torch.utils.data.DataLoader(
		X_val, batch_size=batch_size, shuffle=batch_shuffle,
		drop_last=True)

	previous_z_adjacency_pred = None
	change_adj_count = -1

	def run_batches(
			X_batches, *, train, exhibit=None, prev_z_adjacency_pred=None,
			chge_adj_count=None):
		"""
		Run batches for an epoch

		:param X_batches: torch DataLoader
		:param train: True if training, False if only evaluating
		:param exhibit: class for old means of tracking Latents,
		  as in comments on LATENTS. MATRIX_REPORT, LABEL
		  in code/named_configs/guide_to_config_files.py
		:param prev_z_adjacency_pred: torch tensor or None, adjacency matrix
		 from previous epoch, if any.  Used only at validation phase so the
		 MAP Z is tracked
		:param chge_adj_count: integer the cumulating change adj count
		:return: tuple of:
			mean loss, z_adjacency_pred, theta, prev_z_adjacency_pred,
			chge_adj_count
		"""
		net_call_function = overall_net if train else overall_net.infer
		epoch_samples = 0
		losses = []
		z_adjacency_pred = None
		for X_batch, in X_batches:
			X_pred, z_adjacency_pred, z_pred, theta = net_call_function(
				X_batch)
			if (chge_adj_count is not None) and (
					(previous_z_adjacency_pred is None) or
					(not torch.equal(
						z_adjacency_pred, prev_z_adjacency_pred))):
				prev_z_adjacency_pred = z_adjacency_pred
				chge_adj_count += 1
			loss = torch.mean(torch.square(
				X_pred - X_batch[:, None, :].repeat(1, X_pred.shape[1], 1)),
				dim=(0, 2)
			)
			z_reg_loss = z_regularizer(z_adjacency_pred)
			if train:
				reg_loss = loss + z_reg_loss  # Note can't yet do theta
				# regularization
				optimizer.zero_grad()
				reg_loss.mean().backward()
				if sfe_flag:
					sfe_set_h_grad(
						overall_net,
						z_adjacency_pred.reshape(-1, d ** 2),
						loss,
						noise_temperature
					)
				optimizer.step()
			loss = loss.mean().item()
			b_times_s = X_batch.shape[0] * X_pred.shape[1]
			losses.append(loss * b_times_s)
			epoch_samples += b_times_s
			if (exhibit is not None) and (
					overall_net.mode.mode != 'max_dag_test_only') and (
					not c.VAL_LOSS_FROM_DAG):
				exhibit.update(z_pred)

		if c.WANDB and train:  # Needed to get the grad, which theta above
			# doesn't have
			theta = list(overall_net.h_net.parameters())[0]
		return np.sum(losses) / epoch_samples, z_adjacency_pred, theta, \
			   prev_z_adjacency_pred, chge_adj_count

	best_epoch = None
	min_loss = np.inf
	min_dag_loss = np.inf
	best_z_adjacency_pred = None
	best_metrics = None
	last_metrics = None
	latents_val = LatentExhibit(d) if c.LATENTS else None
	nshd_c_wandb = np.NAN

	# Account for whether c.TQDM = True or False
	def tqdm_null(iter, ncols):
		return iter
	if c.TQDM:
		tqdm_fn = tqdm
	else:
		tqdm_fn = tqdm_null

	# Run through epochs
	for epoch in tqdm_fn(range(1, n_epochs + 1), ncols=50):
		best = False
		overall_net.train()
		train_loss, z_adjacency_pred, train_theta, _, _ = run_batches(
			X_train_batches, train=True)

		wandb_dict = handle_wandb_train(
			d, solver, train_loss, train_theta, z_adjacency_pred)

		if lr_scheduler is not None:
			for lr_scheduler_net in lr_scheduler.values():
				lr_scheduler_net.step()
		overall_net.eval()
		val_loss, z_adjacency_pred, theta, previous_z_adjacency_pred, \
			change_adj_count = run_batches(
				X_val_batches,
				train=False,
				exhibit=latents_val,
				prev_z_adjacency_pred=previous_z_adjacency_pred,
				chge_adj_count=change_adj_count
			)

		best_epoch, best_metrics, last_metrics, min_loss = handle_metrics(
			X_val_batches, adjacency_true, best_epoch, best_metrics,
			change_adj_count, d, epoch, i, latents_val,
			min_dag_loss, min_loss, minizinc_solver, n_epochs,
			overall_net, run_batches, solver, theta, train_loss, trial,
			val_loss, wandbs, z_adjacency_pred, wandb_dict, trial_store)

		if afas_scheduler:
			afas_scheduler.step()

	return last_metrics
