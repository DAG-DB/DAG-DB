"""  Managing torch forward and backward for the DAG-DB framework with use of
Score-function estimation (SFE) as the discrete backprop
"""

import functools
import math

import torch
from torch import Tensor

from discrete_backprop.noise import BaseNoiseDistribution
from discrete_backprop.target import BaseTargetDistribution, TargetDistribution

from typing import Optional, Callable

import logging

import lib.ml_utilities as mlu
from lib.ml_utilities import c
from utils.grad_log import grad_log


logger = logging.getLogger(__name__)


def sfe(function: Optional[Callable[[Tensor], Tensor]] = None,
         target_distribution: Optional[BaseTargetDistribution] = None,
         noise_distribution: Optional[BaseNoiseDistribution] = None,
         nb_samples: int = 1,
         theta_noise_temperature: float = 1.0,
         target_noise_temperature: float = 1.0,
         streamline = False
         ):
    if target_distribution is None:
        target_distribution = TargetDistribution(alpha=1.0, beta=1.0)

    if function is None:
        return functools.partial(sfe,
                                 target_distribution=target_distribution,
                                 noise_distribution=noise_distribution,
                                 nb_samples=nb_samples,
                                 theta_noise_temperature=theta_noise_temperature,
                                 target_noise_temperature=target_noise_temperature)

    @functools.wraps(function)
    def wrapper(theta: Tensor, *args):
        class WrappedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, theta: Tensor, *args):
                """
                Forward call for imle layer

                :param ctx: torch context
                :param theta: (r: = d * (d - 1) / 2 + d ** 2, ) float32 Tensor
                :param args: Not used
                :return: (s, r) float32 Tensor, z
                """
                # theta is (r,)
                r = theta.shape[0]

                # perturbed_theta_shape is (s, r)
                perturbed_theta_shape = [nb_samples, r]

                # ε ∼ ρ(ε)
                # noise is (s, r)
                if noise_distribution is None:
                    noise = torch.zeros(size=perturbed_theta_shape, device=theta.device)
                else:
                    noise = noise_distribution.sample(shape=torch.Size(perturbed_theta_shape)).to(theta.device)

                # eps is (s, r)
                eps = noise * theta_noise_temperature

                # perturbed_theta is (s, r)
                perturbed_theta = theta.view(1, -1).repeat(nb_samples, 1).view(perturbed_theta_shape)
                perturbed_theta = perturbed_theta + eps

                # z is (s, r)
                z = function(perturbed_theta)

                ctx.save_for_backward(theta, noise, z)
                return z

            @staticmethod
            def backward(ctx, dy):
                """
                Backward call for sfe layer - returns zero and the backward
                is dealt with by sfe_set_h_grad

                :param ctx: torch context
                :param dy: (s, r) float32 Tensor
                    r = d * (d - 1) / 2 + d ** 2    lt_p modes
                    r = d ** 2                      max_dag modes
                :return: (r,) float32 Tensor, gradient
                """

                return torch.zeros_like(theta)

        return WrappedFunc.apply(theta, *args)
    return wrapper


def sfe_set_h_grad(overall_net, z, loss_per_sample, noise_temperature):
    """
    Sets theta.grad

    :param z: (s, r) float32 Tensor
    :param loss_per_sample:  (s,) float32 Tensor
    :param noise_temperature:  float
    """

    m_theta = torch.mean(z, dim=0, keepdim=True)

    # Next line is needed as there's no .process() here.  log_prob is (s, r)
    log_prob = (z - m_theta) / noise_temperature

    gradient_sfe = loss_per_sample[:, None] * log_prob

    gradient_sfe = gradient_sfe.mean(dim=0)

    list(overall_net.h_net.parameters())[0].grad = \
            torch.nn.Parameter(gradient_sfe)


if __name__ == "__main__":
    pass
