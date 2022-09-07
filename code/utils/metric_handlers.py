""" Used to handle multiple metrics by code/learn_infer/dag_learner.py
"""


import optuna
from torch import nn

from lib import ml_utilities as mlu
from lib.ml_utilities import c, h
from utils.metrics import count_accuracy
if c.WANDB:
	from utils.wandb_util import insert_to_wandb


dag_every_epoch = hasattr(c, 'DAG_EVERY_EPOCH') and c.DAG_EVERY_EPOCH


class TrialStore:
	"""
	To store trial instructions from hyperparameters during Optuna trails.
	These are replaced in-trial by the associated trial values, but need to
	be restored when trial ends normally (in code/main_dag.py) or when trial
	pruned (below in handle_metrics)
	"""
	def __init__(self):
		self.store = dict()

	def append(self, key):
		self.store[key] = h[key]

	def deploy(self):
		for key, value in self.store.items():
			h[key] = value


def handle_wandb_train(d, solver, train_loss, train_theta, z_adjacency_pred):
	"""
	Update wandb_dict in training

	:param d: int. no. nodes
	:param solver: MAP solver
	:param train_loss: torch scalar tensor
	:param train_theta: torch scalar tensor
	:param z_adjacency_pred: torch tensor (*, d, d), predicted z_adjacency
	 matrix
	:return:
	"""
	wandb_dict = None
	if c.WANDB:
		t_theta_non_adj = train_theta[None, :].reshape(1, d, d) * \
							   (1 - z_adjacency_pred)
		for i in range(d):
			t_theta_non_adj[:, i, i] = 0
		wandb_dict = {
			't_loss': train_loss,
			't_theta': train_theta,
			't_theta_grad': train_theta.grad,
			't_theta_grad_relu': nn.functional.relu(train_theta.grad),
			't_size': z_adjacency_pred.sum(dim=(-2, -1)),
			't_theta_adj': (train_theta[None, :].reshape(1, d, d) *
							z_adjacency_pred),
			't_theta_non_adj': t_theta_non_adj,
			't_calc_thres': solver.calculated_threshold
		}
	return wandb_dict


def handle_metrics(X_val_batches, adjacency_true, best_epoch,
				   best_metrics, change_adj_count, d, epoch, i,
				   latents_val, min_dag_loss, min_loss, minizinc_solver,
				   n_epochs, overall_net, run_batches, solver,
				   theta, train_loss, trial, val_loss, wandbs,
				   z_adjacency_pred, wandb_dict, trial_store):
	"""
	Handle multiple metrics after each epoch.

	Could be trimmed.

	:param X_val_batches:
	:param adjacency_true: numpy (d. d) {0, 1}
	:param best: True if best validation loss so far  ??????
	:param best_epoch: best epoch so far in terms of val_loss
	:param best_metrics: metrics from after best epoch
	:param change_adj_count: count of change between epochs (not intra-epoch)
	:param d: int, no. nodes
	:param epoch: int
	:param i: int, no. of DAG in run
	:param last_metrics: ?????
	:param latents_val: used formerly to track latents
	:param min_dag_loss: min val loss so far after calculating DAG (deprecate?)
	:param min_loss: min val loss so far
	:param minizinc_solver: None or
	code.learn_infer.minizinc_solver.MiniZincSolverFinal instance
	:param n_epochs: total number of epochs for training
	:param overall_net: code.learn_infer.dag_learner.OverallNet instance
	:param run_batches: code.learn_infer.dag_learner.dag_learner.run_batches
	 instance
	:param solver: MAP solver
	:param theta: torch tensor
	:param train_loss: torch scalar tensor
	:param trial: None or Optuna trial instance
	:param val_loss: torch scalar tensor
	:param wandbs: numpy (n_epochs) of dicts with data for wandb by epoch
	:param z_adjacency_pred: torch tensor (*, d, d), predicted z_adjacency
	 matrix
	:param wandb_dict: raw data for wandb from this epoch
	:param trial_store: instance of TrialStore
	:return: best_epoch, best_metrics, last_metrics, min_loss - all updated
	"""
	best = False
	last_metrics = None
	if c.WANDB:
		v_theta_non_adj = theta[None, :].reshape(1, d, d) * \
							   (1 - z_adjacency_pred.detach().cpu().numpy())
		for i in range(d):
			v_theta_non_adj[:, i, i] = 0
		wandb_dict = {
			**wandb_dict,
			'v_loss': val_loss,
			'v_theta': theta,
			'Z': z_adjacency_pred,
			'v_size': z_adjacency_pred.sum(dim=(-2, -1)),
			'v_theta_adj': (theta[None, :].reshape(1, d, d) *
							z_adjacency_pred.detach().cpu().numpy())
			#			   * adj_multiple[:, None, None]
			,
			'v_theta_non_adj': v_theta_non_adj,
			'v_calc_thres': solver.calculated_threshold
		}
	# size = z_adjacency_pred.sum().item()
	if (val_loss < min_loss):
		# if (val_loss < min_loss) and (size >= size_min) and (size <= size_max):
		min_loss = val_loss
		# mlu.save_model(overall_net)
		best = True
	# if True:
	# if best or (epoch == 1) or ((epoch % 10) == 0) or (epoch == n_epochs):
	if (dag_every_epoch or best or (epoch == n_epochs) or (epoch % 100 == 0)) \
			and (
			z_adjacency_pred is not None):
		if overall_net.mode.mode.startswith('max_dag'):
			if (epoch == n_epochs) and (minizinc_solver is not None):
				z_adjacency_pred_orig = z_adjacency_pred.clone()
			if solver.test_only:
				# if overall_net.h_null is not None:
				# 	theta = list(overall_net.h_net.parameters())[
				# 		0].detach().reshape(d, d)
				# else:
				# 	raise NotImplementedError  # Would need theta or X_batch
				if c.VAL_LOSS_FROM_DAG:
					with solver.form_dag_to_infer():
						val_loss, z_adjacency_pred, theta, _, _ \
							= run_batches(
							X_val_batches, train=False,
							exhibit=latents_val)
					# Above, renews theta with z_adjacency_pred
					# as batches will have been shuffled
					if val_loss < min_dag_loss:
						min_dag_loss = val_loss
						best = True
					else:
						best = False
				z_adjacency_pred = solver.test(
					z_adjacency_pred.squeeze(), theta)
		else:
			z_adjacency_pred = z_adjacency_pred.squeeze().detach(
			).cpu().numpy()
		if best:
			best_epoch = epoch
			best_z_adjacency_pred = z_adjacency_pred
		if  dag_every_epoch or best or (epoch == n_epochs) or (epoch % 100 ==
		                                                      0):
			metrics = count_accuracy(adjacency_true, z_adjacency_pred)
			size = metrics['pred_size']
			nshd_c = metrics['nshd_c']
			nshd_c_wandb = nshd_c
			if (i == 0) and ((epoch % 100) == 0) and (trial is not None):
				# See https://github.com/optuna/optuna/issues/276 re
				# pruning intervals
				trial.report(nshd_c, epoch)
				if trial.should_prune():
					trial_store.deploy()
					raise optuna.exceptions.TrialPruned()
			nshd = metrics['nshd']
			tpr = metrics['tpr']
			prec_c = metrics['prec_c']
			metrics = {
				'train_loss': train_loss,
				'val_loss': val_loss,
				'size': size,
				'nshd_c': nshd_c,
				'nshd': nshd,
				'tpr': tpr,
				'prec_c': prec_c,
				'change_adj_count': change_adj_count
			}
			if c.WANDB:
				wandb_dict = {**wandb_dict, **metrics}
				insert_to_wandb(wandbs, d, i, epoch, wandb_dict)
			mlu.log(f' {epoch=:>4}      {train_loss=:9.4f}      '
					f'{val_loss=:9.4f}      {best=:>1}      '
					f'{size=:>4}      '
					f'nSHD_c={nshd_c:9.4f}      '
					f'nSHD={nshd:9.4f}      '
					f'{prec_c=:9.4f}      '
					f'{change_adj_count=}      '
					)

			# wandb.log({
			# 	f'adj_pred': wandb.plots.HeatMap(list(range(d)),
			# 											  list(range(d)),
			# 											  z_adjacency_pred[
			# 											  0, :, :].cpu(
			# 											  ).detach().numpy(),
			# 											  show_text=True),
			# 	# 'adj_true': adjacency_true,
			# 	f'w_pred': wandb.plots.HeatMap(list(range(d)),
			# 											list(range(d)),
			# 											list(
			# 												overall_net.f_net.parameters())[
			# 												0][0].cpu(
			# 											).detach().numpy()
			# 											,
			# 											show_text=False),
			# })
			if (length := len(z_adjacency_pred.shape)) > 2:
				assert length == 3
				z_adjacency_pred = z_adjacency_pred[0]
			mlu.save_to_ods(i, z_adjacency_pred, 'adj_pred')
			if overall_net.h_null is not None:
				weights = list(overall_net.f_net.parameters())[0][
					0].detach().cpu().numpy()
				mlu.save_to_ods(i, weights, 'w_pred')
				mlu.save_to_ods(
					i, weights * z_adjacency_pred, 'w_adj_pred')
		if best:
			best_metrics = metrics
		if epoch == n_epochs:
			if minizinc_solver is not None:
				z_adjacency_pred = \
					minizinc_solver.test(
						z_adjacency_pred_orig.squeeze(), theta)
				metrics = count_accuracy(adjacency_true, z_adjacency_pred)
				size = metrics['pred_size']
				nshd_c = metrics['nshd_c']
				nshd = metrics['nshd']
				# tpr = metrics['tpr']
				prec_c = metrics['prec_c']
				metrics = {
					'train_loss': train_loss,
					'val_loss': val_loss,
					'size': size,
					'nshd_c': nshd_c,
					'nshd': nshd,
					'prec_c': prec_c,
					'change_adj_count': change_adj_count
				}
				mlu.log(f'MiniZinc solver:\t'
						f'{size=:>4}      '
						f'nSHD_c={nshd_c:9.4f}      '
						f'nSHD={nshd:9.4f}      '
						f'{prec_c=:9.4f}      '
						)

			mlu.save_to_ods(i, z_adjacency_pred, f'{str(i).zfill(2)}_adj_pred')

			last_metrics = metrics
	elif (best or (epoch == n_epochs)) and (z_adjacency_pred is None):
		mlu.log(f' {epoch=:>4}      {train_loss=:9.4f}      '
				f'{val_loss=:9.4f}      {best=:>1}'
				)
	return best_epoch, best_metrics, last_metrics, min_loss
