""" From  l2x-aimle/imle/imle.py
Branch:         main
Commit:         c4cca02b41bf2197ee748fd189cb565949160a5e
"""


# -*- coding: utf-8 -*-

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


def d_p_imle(function: Optional[Callable[[Tensor], Tensor]] = None,
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
        return functools.partial(d_p_imle,
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
                :param theta: (r: = 2 * (d ** 2), ) float32 Tensor
                :param args: Not used
                :return: (s, r) float32 Tensor, z
                """
                # theta is (r,)
                r = theta.shape[0]

                # perturbed_theta_shape is (s, r)
                perturbed_theta_shape = [nb_samples, r]

                # noise_shape is (s, d ** 2 + d)
                d = int(math.sqrt(r // 2))
                noise_shape = [nb_samples, d ** 2 + d]

                # ε ∼ ρ(ε)
                # noise is (s, r)
                if noise_distribution is None:
                    noise = torch.zeros(size=noise_shape, device=theta.device)
                else:
                    noise = noise_distribution.sample(
                        shape=torch.Size(noise_shape)).to(theta.device)

                noise_d = noise[:, : d ** 2]
                noise_p = noise[:, d ** 2 :]
                noise_p = (noise_p[:, :, None] - noise_p[:, None, :]).reshape(
                    -1, d ** 2)
                noise = torch.hstack((noise_d, noise_p))

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
                Backward call for imle layer

                :param ctx: torch context
                :param dy: (s, r) float32 Tensor
                    r = d * (d - 1) / 2 + d ** 2    lt_p modes
                    r = d ** 2                      max_dag modes
                :return: (r,) float32 Tensor, gradient
                """

                # theta is (r,)
                # noise is (s, r)
                # z is (s, r)
                theta, noise, z = ctx.saved_tensors

                assert nb_samples == dy.shape[0]

                # target_theta is (s, r)
                # θ' = θ - λ dy
                target_theta = target_distribution.params(
                    theta.unsqueeze(0), dy)

                # eps is (s, r)
                eps = noise * target_noise_temperature

                # perturbed_target_theta is (s, r)
                perturbed_target_theta = target_theta + eps

                # z' = MAP(θ' + ε)
                # z_prime is (s, r)
                z_prime = function(perturbed_target_theta)

                # g = z - z'
                # gradient_sd is (s, r)

                gradient_sr = (z - z_prime)

                # # Streamline overwrite for z_lt
                if streamline:
                    r = theta.shape[0]
                    d = int((1 + math.sqrt(1 + 24 * r)) // 6)
                    mu_diff = torch.sigmoid(
                            theta[: d * (d - 1) // 2])[None, :] - torch.sigmoid(
                                target_theta[:,  :d * (d - 1) // 2])

                    gradient_sr[:, : d * (d - 1) // 2] = \
                        target_distribution.process(
                            theta,  # Not used in the call to TargetDistribution
                            dy,  # Not used in the call to TargetDistribution
                            mu_diff
                        )

                gradient_sr = target_distribution.process(
                    theta, dy, gradient_sr)

                if c.GRAD_LOG:
                    grad_log(gradient_sr[0], '0')

                # # gradient is (r,)
                gradient = gradient_sr.mean(dim=0)

                if c.GRAD_LOG:
                    grad_log(gradient, 'All')

                return gradient

        return WrappedFunc.apply(theta, *args)
    return wrapper


if __name__ == "__main__":
    pass
