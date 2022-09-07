""" Solver for the second alternative architecture of the thesis's sec 5.6
"""

import numpy as np

from utils.base_solver import BaseSolver


class DPSolver(BaseSolver):
	def __call__(self, theta):
		theta_d = theta[:, : self.d ** 2].reshape(-1, self.d, self.d)
		for row in theta_d:
			for i in range(self.d):
				row[i, i] = 0.
		theta_d = theta_d.reshape (-1, self.d ** 2)
		theta_p = theta[:, self.d ** 2 :]

		threshold = self.get_threshold(theta_d)
		theta_d = theta_d * (theta_d > threshold)
		z_d = (theta_d > 0).astype(np.int64)

		z_p = np.argsort(theta_p, axis=1)

		z = np.hstack((z_d, z_p))
		return z
