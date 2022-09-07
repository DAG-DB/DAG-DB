""" Builds the hNet (for Theta) and the fNet for the DAG-DB framework which,
as presented in thesis, uses hNull from the multiple 'h' options.  The key
to constructing fNet is lib.channel_linear.ChannelLinear
"""


import torch
from torch import nn

from lib.channel_linear import ChannelLinear


def build_net(x_width, linear_layers, theta_width, layer_type):
    linear_layers = (32, 32, 32) if linear_layers is None \
        else linear_layers
    linear_layers = (x_width,) + linear_layers + (theta_width,)
    layers = nn.ModuleList([
        layer_type(n_in, n_out)
        for n_in, n_out in zip(linear_layers[: -1], linear_layers[1:])])
    return layers, nn.ReLU()


def d_p_transform(theta, d):
    d_sq = d ** 2
    theta_d = theta[... , : d_sq]
    theta_p = theta[... , d_sq :]
    theta_p = (theta_p[..., :, None] - theta_p[..., None, :])
    new_shape_p = theta_p.shape[:-2] + (d_sq, )
    theta = torch.hstack((theta_d, theta_p.reshape(new_shape_p)))
    return theta


class hNet(nn.Module):
    def __init__(
            self, d, batch_size, linear_layers=None, theta_width=None,
            d_p_flag=False):
        super().__init__()
        self.d = d
        self.batch_size = batch_size
        self.theta_width = theta_width
        self.d_p_flag = d_p_flag
        self.layers, self.relu = build_net(
            d * batch_size, linear_layers, self.theta_width, nn.Linear)

    def forward(self, x):
        x = x.view(-1)
        for layer in self.layers[: -1]:
            x = self.relu(layer(x))
        theta = self.layers[-1](x)
        if self.d_p_flag:
            theta = d_p_transform(theta, self.d)
        return theta


class hNetVectorP(hNet):
    def __init__(self, d, batch_size, device, linear_layers=None):
        super().__init__(
            d, batch_size, linear_layers, theta_width=d * (d - 1) // 2 + d)
        self.layers, self.relu = build_net(
            d * batch_size, linear_layers, self.theta_width, nn.Linear)
        # self.vector_to_matrix = hNetVectorP.sort_vector_to_matrix
        self.vector_to_matrix = self.argsort_vector_to_matrix
        self.range_vector = torch.arange(d, device=device)

    # @staticmethod
    # def sort_vector_to_matrix(vector):
        # sorted = torch.sort(vector).values
        # return torch.outer(sorted, vector)  # Why did outer work?

    def argsort_vector_to_matrix(self, vector):
        argsorted = torch.argsort(vector)
        return torch.outer(argsorted, self.range_vector)

    def forward(self, x):
        theta_vector = super().forward(x)
        theta_vector_p = theta_vector[..., -self.d:]
        theta_matrix_p = self.vector_to_matrix(theta_vector_p)
        theta = torch.cat(
            (theta_vector[..., :-self.d], theta_matrix_p.view(-1)), dim=-1)
        return theta


class hNull(nn.Module):
    def __init__(self, d, bound, theta_width, d_p_flag=False):
        super().__init__()
        self.d = d
        self.bound = bound
        self.theta_width = theta_width
        self.d_p_flag = d_p_flag
        self.theta = nn.Parameter(torch.empty(theta_width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.theta, -self.bound, self.bound)

    def forward(self, x):
        theta = self.theta.clone()
        # if self.d_p_flag:  #
        #     theta = d_p_transform(theta, self.d)
        return theta


class hNullVectorP(hNetVectorP):
    def __init__(self, d, device, bound):
        hNetVectorP.__init__(self, d, 32, device, None)
        self.layers = None
        self.bound = bound
        self.theta = nn.Parameter(torch.empty(self.theta_width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.theta, -self.bound, self.bound)

    def forward(self, x):
        theta_vector = self.theta
        theta_vector_p = theta_vector[..., -self.d:]
        theta_matrix_p = self.vector_to_matrix(theta_vector_p)
        theta = torch.cat(
            (theta_vector[..., :-self.d], theta_matrix_p.view(-1)), dim=-1)
        return theta


class fNet(nn.Module):
    def __init__(self, d, s, device, mode, linear_layers=None, f_bias=True):
        super().__init__()
        self.d = d
        self.s = s
        self.device = device
        self.lt_width = d * (d - 1) // 2
        self.p_width = d ** 2
        self.tril_indices = torch.tril_indices(d, d, offset=-1, device=device)
        match mode.mode:
            case _ if mode.mode.startswith('lt_p'):
                self.get_z_adjacency = self.get_z_adjacency_lt_p
            case _ if mode.mode.startswith('max_dag'):
                self.get_z_adjacency = self.get_z_adjacency_max_dag
            case 'd_p':
                self.get_z_adjacency = self.get_z_adjacency_d_p
            case _:
                raise NotImplementedError
        self.z_adjacency_pred = None

        def d_channel_linear(n_in, n_out):
            return ChannelLinear(d, f_bias, n_in, n_out)

        self.layers, self.relu = build_net(
            d, linear_layers, 1, d_channel_linear)

    def to_matrix_lower_triangle(self, z_lt):
        """

        :param z_lt: (s, d * (d - 1) / 2) float32 Tensor
        :return: (s, d, d) float32 Tensor, lower triangular matrix rep of z_lt
        """
        matrix_z_lt = torch.zeros(
            (z_lt.shape[0], self.d, self.d),
            dtype=torch.float32,
            device=self.device
        )
        matrix_z_lt[
            :, self.tril_indices[0], self.tril_indices[1]] = z_lt
        return matrix_z_lt

    def get_z_adjacency_lt_p(self, z):
        """
        Get z in its representation as adjacency matrix

        z_adjacency[i, j]  = sum_{r, s}
         KroneckerDelta[r, perm_z_p[i]] * z_lt[r, s]
          * KroneckerDelta[s, perm_z_p[j]]

        :return: (s, d, d) float32 Tensor, z_adjacency
        """
        z_lt = self.to_matrix_lower_triangle(z[:, :self.lt_width])
        z_p = z[:, -self.p_width:].view(-1, self.d, self.d)
        return z_p.transpose(-2, -1) @ z_lt @ z_p

    def get_z_adjacency_max_dag(self, z):
        return z.reshape(-1, self.d, self.d)

    def get_z_adjacency_d_p(self, z):
        z_d = z[:, : self.d ** 2]
        z_p = z[:, self.d ** 2 :]
        z_p = (z_p[:, :, None] > z_p[:, None, :]).reshape(-1, self.d ** 2)
        z =  (z_d * z_p).reshape(-1, self.d, self.d)
        return z

    def forward(self, x, z):
        """

        :param x: (b, d) float32 Tensor, the data
        :param z: (s, d * (d -1) /2 + d ** 2) float32 Tensor
        :return: (b, d) float32 Tensor, the prediction of x
        """
        z_adjacency = self.get_z_adjacency(z)
        self.z_adjacency_pred = z_adjacency

        # See top of file for adjacency convention.  The next line returns
         # an (b, s) array of modified adjacency matrices,
         # the modification being that
         #    1 at z_adjacency[b, s, i, j] is replaced by x_roots[b, s, i],
         # which gives effect to i being the parent of j.
        # x_roots = self.restrict_to_roots(x, z_adjacency)
        x = z_adjacency[None, :, :, :] * x[:, None, :, None]
        for layer in self.layers[: -1]:
            x = self.relu(layer(x))
        x = self.layers[-1](x).squeeze(-2)  # squeeze from shape
         # (b, s, 1, d)
        return x, z_adjacency  # (b, d)
