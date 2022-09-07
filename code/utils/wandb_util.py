""" Utilities to support used of wandb
"""


import numpy as np
import torch

from lib.ml_utilities import c
import lib.ml_utilities as mlu

if c.WANDB:
	import wandb

	def insert_to_wandb(wandbs, d, i, epoch, wandb_dict):
		"""
		Update wandb data

		:param wandbs: wandb data, a numpy (N_EPOCHS,) array of dict
		:param d: int, no. of nodes
		:param i: int, no. of DAG in this run
		:param epoch: int, epoch no.
		:param wandb_dict: raw data for wandb from this epoch
		:return: wandbs
		"""
		for key in wandb_dict:
			if isinstance(wandb_dict[key], torch.Tensor):
				wandb_dict[key] = wandb_dict[key].detach().cpu().numpy()

		t_size = np.clip(wandb_dict['t_size'], 1, d * (d - 1) - 1)[
				 :, None, None]
		v_size = np.clip(wandb_dict['v_size'], 1, d * (d - 1) - 1)[
				 :, None, None]

		wandb_dict['t_theta_adj'] *= (d ** 2) / t_size
		wandb_dict['t_theta_non_adj'] *= (d ** 2) / (d * (d - 1) - t_size)
		wandb_dict['v_theta_adj'] *= (d ** 2) / v_size
		wandb_dict['v_theta_non_adj'] *= (d ** 2) / (d * (d - 1) - v_size)

		new_wandbs = {
			f'{key}[{i}]': value for key, value in wandb_dict.items()}
		for key, value in new_wandbs.items():
			if not (isinstance(value, float) or isinstance(value, int)):
				new_wandbs[key] = value.mean()

		wandbs[epoch - 1] = {
			**wandbs[epoch - 1],
			**new_wandbs
		}
		return wandbs


	def finalise_wandb(wandbs):
		""" When training finishes send wandbs to wandb
		"""
		for epoch, elt in enumerate(wandbs, 1):
			wandb.log(elt, step=epoch)
		wandb.config.log_file = mlu.LOG_FILESTEM
