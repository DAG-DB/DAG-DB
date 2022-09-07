"""  Managing torch forward and backward for the DAG-DB framework with use of
straight-through estimation (STE) as the discrete backprop.  With thanks to
Pasquale Minervini from whose I-MLE repo this is adapted
"""

import functools

import torch
from torch import Tensor

from discrete_backprop.noise import BaseNoiseDistribution
from discrete_backprop.target import BaseTargetDistribution, TargetDistribution

from typing import Optional, Callable

import logging

logger = logging.getLogger(__name__)


def ste(function: Optional[Callable[[Tensor], Tensor]] = None,
         target_distribution: Optional[BaseTargetDistribution] = None,
         noise_distribution: Optional[BaseNoiseDistribution] = None,
         nb_samples: int = 1,
         theta_noise_temperature: float = 1.0,
         target_noise_temperature: float = 1.0,
        streamline=None
         ):
    if target_distribution is None:
        target_distribution = TargetDistribution(alpha=1.0, beta=1.0)

    if function is None:
        return functools.partial(ste,
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
                :param theta: (r: = d * (d + 1) / 2 + d, ) float32 Tensor
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
                Backward call for imle layer

                :param ctx: torch context
                :param dy: (s, r: = d * (d + 1) / 2 + d) float32 Tensor
                :return: (r,) float32 Tensor, gradient
                """
                # divide by temperature (from I-MLE eq 7 with tau != 1) as
                # theres' no .process() stage to do this
                gradient = dy.mean(dim=0) / theta_noise_temperature

                return gradient

        return WrappedFunc.apply(theta, *args)
    return wrapper


if __name__ == "__main__":
    pass
