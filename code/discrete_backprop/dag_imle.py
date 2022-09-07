"""  Managing torch forward and backward for the DAG-DB framework with use of
I_MLE as the discrete backprop.  With thanks to Pasquale Minervini from whose
I-MLE code this is adapted
"""


import functools
from typing import Optional, Callable

import torch
from torch import Tensor

from discrete_backprop.noise import BaseNoiseDistribution
from discrete_backprop.target import BaseTargetDistribution, TargetDistribution
from lib.ml_utilities import c
from utils.grad_log import grad_log


def imle(function: Optional[Callable[[Tensor], Tensor]] = None,
         target_distribution: Optional[BaseTargetDistribution] = None,
         noise_distribution: Optional[BaseNoiseDistribution] = None,
         nb_samples: int = 1,
         theta_noise_temperature: float = 1.0,
         target_noise_temperature: float = 1.0,
         streamline = False
         ):
    # Set up the function
    if target_distribution is None:
        target_distribution = TargetDistribution(alpha=1.0, beta=1.0)

    if function is None:
        return functools.partial(imle,
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

                if streamline:  # NB works only for max_dag modes
                    gradient_sr = (torch.sigmoid(
                            theta)[None, :] - torch.sigmoid(target_theta)
                        )
                else:
                    # eps is (s, r)
                    eps = noise * target_noise_temperature

                    # perturbed_target_theta is (s, r)
                    perturbed_target_theta = target_theta + eps

                    # z' = MAP(θ' + ε)
                    # z_prime is (s, r)
                    z_prime = function(perturbed_target_theta)

                    # g = z - z'
                    # gradient_sd is (s, r)
                    gradient_sr = z - z_prime

                gradient_sr = target_distribution.process(
                    theta, dy, gradient_sr)  # divides by (\lambda \tau)

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
