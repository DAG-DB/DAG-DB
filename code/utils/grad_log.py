""" Log gradient-related information
"""

import math

import torch

import lib.ml_utilities as mlu


def grad_log(gradient, tag):
	"""
	Log gradient-related information

	:param gradient: float to log
	:param tag: string for short description of gradient
	:return:
	"""
	d = int(math.sqrt(gradient.shape[-1]))
	gradient = gradient.reshape(-1, d, d)
	for i in range(d):
		gradient[:, i, i] = 0
	gradient = gradient.reshape(-1, d ** 2)

	values, counts = torch.unique(gradient, return_counts=True)
	values = [value.item() for value in values]
	counts = [count.item() for count in counts]
	grad_count = list(zip(values, counts))
	mlu.log(f'{tag}:\t{grad_count=}')
	mlu.log(f'\t\t{gradient.mean().item()=}')
