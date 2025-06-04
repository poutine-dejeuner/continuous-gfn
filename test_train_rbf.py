import numpy as np
import torch
import matplotlib.pyplot as plt

from icecream import ic

from tuto import action_parameters_to_state_space

"""
    Il faut s'assurer que les centres des gaussiennes et la grille d'evaluation
    des RBF sont alignés.
    - on fixe l'intervalle dans lequel les points de la grille sont (genre
      (-1,1) ou (0,1)) et on normalise les centres des gaussiennes dans cet
      intervalle
    - les points de la grille sont torch.arange et on normalise les centres
      avec le max de torch.arange

      les 2 reviennent au meme...
"""

np.set_printoptions(precision=2)

class RBFmodel(torch.nn.Module):
    def __init__(self, n_images, n_gaussians, im_shape):
        super(RBFmodel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'cpu')
        self.dtype= torch.float
        self.n_images = n_images 
        self.n_gaussians = n_gaussians
        self.im_shape = im_shape
        self.centers = torch.nn.Parameter(torch.rand(n_images, n_gaussians, 2))
        self.covariances = torch.nn.Parameter(torch.rand(n_images, n_gaussians, 3))
        self.amplitudes = torch.nn.Parameter(torch.randn(n_images, n_gaussians, 1))
        
        grille = self.grid()
        assert self.centers[...,0].min() >= grille[...,0].min() 
        assert self.centers[...,0].max() <= grille[...,0].max()
        assert self.centers[...,1].min() >= grille[...,1].min() 
        assert self.centers[...,1].max() <= grille[...,1].max()

    def forward(self):
        out = rbf(self.centers, self.covariances, self.amplitudes, self.grid(), image_shape=self.im_shape)
        return out

    def grid(self):
        rows, cols = self.im_shape
        return grille(rows, cols).to(self.device, dtype=self.dtype)

def grille(rows, cols):
    x = torch.linspace(-1, 1, rows)
    y = torch.linspace(-1, 1, cols)

    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid_points = torch.stack((X.flatten(), Y.flatten()), dim=-1)
    n_gridpoints = grid_points.shape[0]

    # grille avec arange plutot que linspace
    # r = torch.arange(rows)
    # c = torch.arange(cols)
    # r_grid, c_grid = torch.meshgrid(r, c, indexing='ij')
    # return c_grid, r_grid
    return grid_points

def rbf(centers, covariances, amplitudes, grid_points, image_shape=(101,91)):                                              
    assert grid_points.shape[0] == image_shape[0] * image_shape[1]
    dtype = torch.float
    paramtype = type(centers)
    device = centers.device
    parameters = centers.to(device)

    # Unpack parameters
    # centers = parameters[..., :, :2]
    # centers[...,0] = centers[...,0]/image_shape[0]
    # centers[...,1] = centers[...,1]/image_shape[1]
    # covariances = parameters[..., :, 2:5]
    # amplitudes = parameters[..., :, 5]
    # covariance_mat is an array of shape (N, 2, 2)
    covariance_mat = torch.zeros(parameters.shape[:-1] +  (2, 2), device=device)
    covariance_mat[..., 0, 0] = covariances[..., 0]
    covariance_mat[..., 1, 1] = covariances[..., 1]
    covariance_mat[..., 0, 1] = covariances[..., 2]
    covariance_mat[..., 1, 0] = covariances[..., 2]

    # grid_points = grille(image_shape[0], image_shape[1], dtype=dtype,
                         # device=device)
    n_gridpoints = grid_points.shape[0]
    # grid_points is an array of shape (N, height * width, 2)
    # Compute the difference from the centers
    # centers = centers.unsqueeze(1)
    # centers = centers.repeat(1, num_gridpoints, 1)
    centers = centers.unsqueeze(-2)
    centers = torch.repeat_interleave(centers, repeats=n_gridpoints, dim=-2)
    ic(grid_points.device, centers.device)
    diff = grid_points - centers
    z = torch.einsum('...ij,...jk,...ik->...i', diff, covariance_mat, diff)
    # det = torch.linalg.det(covariance_mat).unsqueeze(-1).expand_as(z)
    # z = z/(2*torch.pi*torch.sqrt(det))

    # take the diagonal in dims -2, -1
    z = torch.exp(-0.5 * z)
    # multiply by the amplitude
    amplitudes = amplitudes.unsqueeze(-1).expand_as(z)
    z = z * amplitudes
    z = z.sum(-2)
    z = z.reshape(z.shape[:-1] + (image_shape[1], image_shape[0]))
    z = torch.transpose(z, -1, -2)
    if paramtype is np.ndarray:
        z = z.numpy()
    return z

def plot(images, N=2):
    _, axs = plt.subplots(N, N, figsize=(15, 5))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i])
        ax.axis('off')
    plt.show()


def test__rbf():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rbfmod = RBFmodel(n_images=4, n_gaussians=3, im_shape=(101, 91))
    rbfmod.to(device)
    ic(next(rbfmod.parameters()).device)
    with torch.no_grad():
        im = rbfmod.forward().numpy()
    plot(im) 


test__rbf()
