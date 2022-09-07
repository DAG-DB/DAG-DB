""" Used only for discarded alternative architecture (the first mentioned
in the thesis' sec 5.6
"""

from contextlib import contextmanager

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment

from utils.base_solver import BaseSolver
""" Scheme for permutations:
    for a permutation which takes x into y:
       perm_matrix:
            (d, d) float32 ndarray, where (d) is the shape of theta and
             y = x @ perm,
             so matrix[i, j] = KroneckerDelta[i, perm[j]].
        else:
            (d) int64 ndarray, where the ith entry is the x-index of the ith
             element of y (ith starting from 0th), so y[i] = x[perm(i)].
             E.g. for 
             >>> x = np.array([
             >>>     2.03039883, 3.99560666, 0.01979908, 0.98609382, 2.95560246
             >>> ])
             >>> y = np.array([
             >>>     0.01979908, 0.98609382, 2.03039883, 2.95560246, 3.99560666
             >>> ])
             the permutation is
             >>> [2 3 0 4 1]
"""


class DAGSolver(BaseSolver):
    def __init__(self, d, s, max_size=None, threshold=0.0, mode=None):
        super().__init__(d, s, max_size, threshold, mode)
        self.lt_width = d * (d - 1) // 2
        self.p_width = d if mode.mode == \
                            'lt_p_vector_argsort_in_std_autograd' else d ** 2
        self.tril_indices = np.tril_indices(d, k=-1)

    def to_matrix_lower_triangle(self, vector_theta_lt):
        """

        :param vector_theta_lt: (s, d * (d - 1) / 2) float32 ndarray
        :return:
        """
        s = vector_theta_lt.shape[0]
        matrix_theta_lt = np.zeros((s, self.d, self.d), dtype=np.float32)
        matrix_theta_lt[
            :, self.tril_indices[0], self.tril_indices[1]] = vector_theta_lt
        return matrix_theta_lt

    def perm_from_vector_to_matrix(self, perm_vector):
        perm_matrix = np.zeros((self.s, self.d, self.d), dtype=np.int64)
        np.add.at(
            perm_matrix,
            (self.s_indices, self.row_indices, perm_vector.flatten()),
            1
        )
        return perm_matrix

    @staticmethod
    def dag_linear_sum_assignment(matrix):
        row_indices, col_indices = scipy_linear_sum_assignment(
            matrix, maximize=True)
        return col_indices

    def __call__(self, theta):
        """

        :param theta: (s, d * (d - 1) / 2 + d ** 2) float32 ndarray
        :return: (s. d * (d - 1) / 2 + d ** 2) int64 ndarray, z
        """
        theta_lt =  theta[:, : self.lt_width]
        threshold = self.get_threshold(theta_lt)
        z_lt = (theta_lt > threshold).astype(np.int64)

        if self.mode == 'lt_p_vector_argsort_in_std_autograd':
            theta_p = theta[:, -self.p_width:]
            z_p = self.perm_from_vector_to_matrix(np.vstack(
                [np.argsort(elt) for elt in list(theta_p)]))
        else:
            theta_p = theta[:, -self.p_width:].reshape(-1, self.d, self.d)
            z_p = self.perm_from_vector_to_matrix(np.vstack(
                [DAGSolver.dag_linear_sum_assignment(elt)
                   for elt in list(theta_p)]
            ))

        z = np.hstack((z_lt, z_p.reshape(self.s, self.d ** 2)))
        return z


if __name__ == "__main__":
    dag_solver = DAGSolver(4, 2, max_size=3)
    theta_lt = np.array([
        [0.5, -1.5, 2.5, 0.2, 1.2, 0.8],
        [0.5, 1.5, 2.5, 0.2, 0.6, 0.8],
    ])
    theta_p = np.array([
        [
            [0.4, -1.5, 2.5, 1.9],
            [0.5, 1.5, 3.5, 8.1],
            [0.2, 1.2, 0.8, -0.4],
            [0.2, 1.2, -7.21, -0.4],
        ],
        [
            [3.2, 0.2, 1.5, -2.6],
            [-1.4, -0.5, -1.5, -.5],
            [3.1, 2.6, 1.2, 0.8],
            [-2.1, -1.9, -3.4, 0.8],
        ]
    ]).reshape((-1, 4 ** 2))
    theta = np.hstack((theta_lt, theta_p))
    z = dag_solver(theta)
    z_lt = z[:, :6]
    pass

    # z_p = dag_solver(theta_p)
    # pass
