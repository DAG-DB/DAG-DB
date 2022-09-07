""" Calculate sizes (no. of edges) of ER and SF graph types
"""

from scipy.stats import binom


def er_size(d, k, proportion):
	"""
	For ER graphs

	:param d: int, no. of nodes
	:param k: int, as in ERk
	:param proportion: float between 0 and 1, the quantile of the size
	 distribution to calculate
	:return: the size which represents the quantile proportion for ERk
	 graphs with d nodes
	"""
	p = 2 * k / (d - 1)
	pt = binom.ppf(proportion, d * (d - 1) / 2, p)
	return int(pt)


def sf_size(d, k, offset=0, multiplier=1.):
	"""
	For SF graphs

	:param d: int, no. of nodes
	:param k: int, as in SFk
	:param offset: int or float, offset to add to calculated size
	:param multiplier: float, multiplier to calculated size
	:return: the size of an SFk graph with d nodes, adjusted by offset and
	 multiplier
	"""
	size = k * (2 * d - k - 1) // 2
	return int(size * multiplier + offset)


if __name__ == '__main__':
	print(sf_size(10, 2))
