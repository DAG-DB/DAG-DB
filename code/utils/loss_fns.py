""" Used for experiments with discarded loss functions
"""


import math
from abc import ABC, abstractmethod

import numpy as np
from scipy.special import lambertw
import torch


class GOLEMLoss(ABC):
    def __init__(self, overall_net):
        self.overall_net = overall_net
        self.f_net = overall_net.f_net

    def get_B(self):
        return list(self.f_net.parameters())[0][0][
            None, :] * self.f_net.z_adjacency_pred.clone()
            # .clone() is needed for when self.f_net.z_adjacency_pred is
        # produced in inference mode

    def get_logdet_term(self):
        B = self.get_B()
        return - torch.slogdet(
            torch.eye(self.overall_net.d, device=B.device) - B).logabsdet.sum()

    @abstractmethod
    def __call__(self, x_pred, x, ):
        raise NotImplementedError


class GOLEMLossNV(GOLEMLoss):
    def __call__(self, x_pred, x):
        """
        From GOLEM paper 'On the role of sparsity and DAG constraints for
        learning linear DAGs', Ng+, 2020.  Eq. above (2).  The log term is
        automatically 0 IF we produce only DAGs

        Also include a final normalisation term (with "-torch.log(..)") to allow
        comparison with X @ B in main_dag.py

        :param x_pred: (b, s, d) float32 Tensor
        :param x: (b, s, d) float32 Tensor
        :return: float32 torch scalar
        """

        # Keep d dimension separate
        x = x.flatten(end_dim=-2)
        x_pred = x_pred.flatten(end_dim=-2)
        return (torch.log(torch.square(x - x_pred).sum(dim=0)).sum(
            ) - x.shape[-1] * math.log(x.shape[0])) / 2 + \
            self.get_logdet_term()


class GOLEMLossEV(GOLEMLoss):
    def __call__(self, x_pred, x):
        """
        From GOLEM paper 'On the role of sparsity and DAG constraints for
        learning linear DAGs', Ng+, 2020.  Eq. (2), based on equal variance
        assumptions.

        Also include a final normalisation term (with "-torch.log(..)") to allow
        comparison with X @ B in main_dag.py

        :param x_pred: (b, s, d) float32 Tensor
        :param x: (b, s, d) float32 Tensor
        :return: float32 torch scalar
        """

        # Keep d dimension separate
        d = x.shape[-1]
        b_times_s = x.shape[0] * x.shape[1]
        return (torch.log(torch.square(x - x_pred).sum()) - math.log(
            b_times_s)) *  d / 2 + \
            self.get_logdet_term()


def er4_log_prob(z_pred, d):
    z_pred = z_pred.to(torch.float32)
    er = 4
    p = er / (d - 1)
    max_edges = d * (d - 1) // 2
    size_on = z_pred[..., :max_edges].sum(dim=-1)
    size_off = max_edges  - size_on
    return (size_on * math.log(p) + size_off * math.log(1 - p)).mean()


class Golem1ER4Loss:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def __call__(self, x_pred, x, z_pred):
        return golem_loss_1(x_pred, x) + self.alpha * er4_log_prob(
            z_pred, x.shape[-1])


class MeanSquareSigmoidRegularizer:
    def __init__(self, d, device, alpha, beta):
        self.d = d
        self.theta_lt_width = d * (d - 1) // 2
        self.device = device
        self.alpha = alpha
        self.beta = beta

    def __call__(self, theta):
        if isinstance(theta, np.ndarray):
            theta = torch.tensor(
                theta, dtype=torch.float32, device=self.device)
        return self.alpha * torch.mean(torch.square(torch.sigmoid(
                theta[..., :self.theta_lt_width] - self.beta)))


class NullRegularizer:
    def __init__(self):
        pass

    def __call__(self, theta):
        return 0
