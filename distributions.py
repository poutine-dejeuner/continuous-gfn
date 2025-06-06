from typing import Optional
import torch
from torch.distributions import (Beta, MultivariateNormal, Normal,
        Distribution, Independent, constraints)
import matplotlib.pyplot as plt
from icecream import ic


class BetaNormal(Distribution):
    arg_constraints = {
        "beta_concentration1": constraints.positive,
        "beta_concentration0": constraints.positive,
        "normal_mean": constraints.real,
    }
    support = constraints.real
    has_rsample = False

    def __init__(self, beta_concentration1, beta_concentration0, normal_mean,
            normal_cov_root, validate_args=None):
        """
        beta_concentration1: (..., M)
        beta_concentration0: (..., M)
        normal_mean: (..., N)
        normal_cov_root: (..., N, N)
        """
        assert (normal_mean.shape[-1] == normal_cov_root.shape[-1] ==
            normal_cov_root.shape[-2]), f"""{normal_cov_root.shape}"""

        self.beta_concentration1 = beta_concentration1
        self.beta_concentration0 = beta_concentration0

        self.normal_mean = normal_mean
        self.normal_cov_root = normal_cov_root
        ic(normal_cov_root.shape)
        # ic(normal_cov_root)
        ic(normal_cov_root[0])
        input()
        # produit de matrices
        self.normal_cov = torch.einsum("...ij, ...jk->...ik", normal_cov_root,
                                            normal_cov_root.transpose(-2, -1))
        ic(self.normal_cov[0])

        self.beta = Beta(beta_concentration1, beta_concentration0)
        self.multinormal = MultivariateNormal(
            self.normal_mean, self.normal_cov)
        super().__init__(validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        beta_sample = self.beta.sample(sample_shape)
        normal_sample = self.multinormal.sample(sample_shape)
        return torch.cat([beta_sample, normal_sample], dim=-1)

    def log_prob(self, value):
        beta_val = value[..., :self.beta.event_shape[0]]
        normal_val = value[..., self.beta.event_shape[0]:]
        return self.beta.log_prob(beta_val) + self.multinormal.log_prob(normal_val)


def test__BetaNormal():
    dist = BetaNormal(
        beta_concentration1=torch.tensor([2.0, 2.0]),
        beta_concentration0=torch.tensor([2.0, 2.0]),
        normal_mean=torch.randn(4),
        normal_cov_root=torch.randn(4, 4)
    )
    x = dist.sample()
    print(x.shape)


def plot_hist():
    N = 4
    _, axes = plt.subplots(N, N)
    axes = axes.flatten()
    for i in range(N**2):
        b1 = torch.randn(1)**2
        b0 = torch.randn(1)**2
        dist = Beta(b1, b0)
        n_samples = int(1e4)
        s = dist.sample((n_samples,)).numpy()
        axes[i].hist(s, bins=100)
        axes[i].axis('off')
        axes[i].set_title("{0:.2f}".format(b1.item()) +
                          ", " + "{0:.2f}".format(b0.item()))
    plt.savefig('test_hist.png')
    plt.show()

def batch_matmul(A, B):
    out = torch.einsum("...ij, ...jk->...ik", A, B)
    return out

class RBF():
    """
        une classe qui contient les parametres d'une rbf
    """
    def __init__(self, params:torch.Tensor):
        """
        params: (...,7)
        """
        assert params.shape[-1] == 7
        if params.ndim == 1:
            params = params.unsqueeze(0)
        self.device = params.device
        self.dtype = params.dtype
        self.centers = params[..., :2]
        self.amplitudes = params[..., 2]
        cov = params[..., 3:]
        cov = cov.reshape(*cov.shape[:-1],2,2)
        self.cov_mat = batch_matmul(cov, cov.transpose(-2,-1))

    def __call__(self, grid_points):
        centers = self.centers
        diff = broadcast_add(grid_points, -centers)
        cov_mat = self.cov_mat
        z = torch.einsum('...ij,...ijk,...ik->...i', diff, cov_mat, diff)
        z = torch.exp(-0.5 * z)
        z = z * self.amplitudes
        z = z.sum(-1)
        return z

def broadcast_add(A, B):
    a_shape, b_shape = A.shape[:-1], B.shape[:-1]
    d = A.shape[-1]
    A_exp = A.reshape(a_shape + (1,) * len(b_shape) + (d,))
    B_exp = B.reshape((1,) * len(a_shape) + b_shape + (d,))
    return A_exp + B_exp

def grid(image_shape:tuple, xrange:tuple=None, yrange:tuple=None):
    x = torch.linspace(*xrange, image_shape[0])
    y = torch.linspace(*yrange, image_shape[1])
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid_points = torch.stack((X, Y), dim=-1)
    return grid_points

def test__rbf():
    im_shape = (101, 91)
    grid_pts = grid(im_shape, (-1, 1), (-1, 1))

    if True:
        print("test single gaussian")
        dist = BetaNormal(
                torch.randn((2,))**2,
                torch.randn((2,))**2,
                torch.randn((5,)),
                torch.randn((5,5)),
                )
        p = dist.sample()
        rbf = RBF(p)
        im = rbf(grid_pts)
        assert im.shape == im_shape, f"{im.shape}"
        plt.imshow(im.numpy())
        plt.axis('off')
        plt.savefig('rbftest1.png')
    if True:
        print("test several gaussian")
        n_fun = 4
        dist = BetaNormal(
                torch.randn((n_fun, 2,))**2,
                torch.randn((n_fun, 2,))**2,
                torch.randn((n_fun, 5,)),
                torch.randn((n_fun, 5,5)),
                )
        p = dist.sample()
        rbf = RBF(p)
        im = rbf(grid_pts)
        assert im.shape == im_shape, f"{im.shape}"
        plt.imshow(im.numpy())
        plt.axis('off')
        plt.savefig('rbftest2.png')
    if True:
        from math import sqrt
        print("test several images several gaussians")
        n_im = 4 
        n_fun = 3 
        dist = BetaNormal(
                torch.randn((n_im, n_fun, 2,))**2,
                torch.randn((n_im, n_fun, 2,))**2,
                torch.randn((n_im, n_fun, 5,)),
                torch.randn((n_im, n_fun, 5,5)),
                )
        p = dist.sample()
        rbf = RBF(p)
        im = rbf(grid_pts)
        assert im.shape[:2] == im_shape, f"{im.shape}"
        assert im.shape[-1] == n_im, f"{im.shape}"

        n = int(sqrt(n_im))
        _, axes = plt.subplots(n, n)
        axes = axes.flatten()
        for i in range(n_im):
            axes[i].imshow(im[...,i].numpy())
            axes[i].axis('off')
        plt.savefig('rbftest3.png')



if __name__ == "__main__":
    test__rbf()
    # test__BetaNormal()
    # plot_hist()
