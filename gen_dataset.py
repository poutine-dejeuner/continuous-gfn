import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

def normalise(image):
    return (image - image.min()) / (image.max() - image.min())

def rbf(centers, covariances, amplitudes, image_shape=(101,91)):
    assert centers.shape[0:2] == covariances.shape[0:2] == amplitudes.shape[0:2]

    dtype = torch.float
    centers = type(centers)
    device = centers.device

    covariance_mat = torch.zeros(covariances.shape[:-1] +  (2, 2), device=device)
    covariance_mat[..., 0, 0] = covariances[..., 0]
    covariance_mat[..., 1, 1] = covariances[..., 1]
    covariance_mat[..., 0, 1] = covariances[..., 2]
    covariance_mat[..., 1, 0] = covariances[..., 2]

    # Create a grid of points
    x = np.linspace(0, 1, image_shape[0])
    y = np.linspace(0, 1, image_shape[1])

    X, Y = np.meshgrid(x, y)
    grid_points = np.stack((X.flatten(), Y.flatten()), axis=-1)
    n_gridpoints = grid_points.shape[0]
    grid_points = torch.tensor(grid_points, dtype=dtype, device=device)

    centers = centers.unsqueeze(-2)
    centers = torch.repeat_interleave(centers, repeats=n_gridpoints, dim=-2)
    diff = grid_points - centers
    z = torch.einsum('...ij,...jk,...ik->...i', diff, covariance_mat, diff)
    z = torch.exp(-0.5 * z)
    amplitudes = amplitudes.unsqueeze(-1).expand_as(z)
    z = z * amplitudes
    z = z.sum(-2)
    z = z.reshape(z.shape[:-1] + (image_shape[1], image_shape[0]))
    z = torch.transpose(z, -1, -2)
    return z


class RBFImageCollection(nn.Module):
    def __init__(self, num_images, num_rbf, image_shape,
                 centers:torch.Tensor=torch.empty(0),
                 covariances:torch.Tensor=torch.empty(0),
                 amplitudes:torch.Tensor=torch.empty(0)):
        super().__init__()
        self.num_rbf = num_rbf
        self.image_shape = image_shape
        rows, cols = image_shape
        if centers.numel() != 0:
            self.device = centers.device
            assert centers.shape[0] == covariances.shape[0] == amplitudes.shape[0], \
                f"Centers shape: {centers.shape}, " \
                f"Covariances shape: {covariances.shape}, " \
                f"Amplitudes shape: {amplitudes.shape}"

            assert centers.ndim == amplitudes.ndim == 3 and covariances.ndim == 4, \
                f"Centers ndim: {centers.ndim}, " \
                f"Covariances ndim: {covariances.ndim}, " \
                f"Amplitudes ndim: {amplitudes.ndim}"

            self.num_images = centers.shape[0]
            self.num_rbf = centers.shape[1]
            self.centers = centers
            self.covariances = covariances
            self.amplitudes = amplitudes
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else
                                       "cpu")
            self.centers = nn.Parameter(torch.randn(num_images, num_rbf, 2))
            self.covariances= nn.Parameter(torch.rand(num_images, num_rbf, 3))
            self.amplitudes = nn.Parameter(torch.randn(num_images, num_rbf, 1))

        if self.num_rbf > self.centers.shape[1]:
            N = self.num_rbf - self.centers.shape[1]
            new_centers = torch.randn(num_images, N, 2, device=self.device)
            new_covariances = torch.rand(num_images, N, 3, device=self.device)
            new_amplitudes = torch.randn(num_images, N, 1, device=self.device)
            self.centers = torch.cat((self.centers, new_centers), dim=1)
            self.covariances = torch.cat((self.covariances, new_covariances),
                                         dim=1)
            self.amplitudes = torch.cat((self.amplitudes, new_amplitudes),
                                        dim=1)

        self.grid = self.make_grid(rows, cols)

    def rbf_function(self, coords):
        x, y = coords
        value = torch.zeros_like(x, dtype=torch.float32)
        for i in range(self.num_rbf):
            cx, cy = self.centers[i]
            sigma = torch.exp(self.log_widths[i])
            w = self.weights[i]
            value += w * \
                torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        return value + self.bias

    def make_grid(self, rows, cols):
        r = torch.arange(rows, dtype=torch.float32)
        c = torch.arange(cols, dtype=torch.float32)
        r_grid, c_grid = torch.meshgrid(r, c, indexing='ij')
        grid = (c_grid.to(self.device), r_grid.to(self.device))
        return grid

    def forward(self):
        output = self.rbf_function(self.grid)
        return torch.sigmoid(output)


def optimize_rbf_coll(target_images_np, num_rbf=20, learning_rate=0.01, num_epochs=5000):
    assert target_images_np.min() >= 0 and target_images_np.max() <= 1

    image_shape = target_images_np[0].shape
    num_images = target_images_np.shape[0]
    target_images_torch = torch.tensor(np.array(
        target_images_np), dtype=torch.float32).unsqueeze(1)  # [batch_size, C, H, W]

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RBFImageCollection(num_images, num_rbf, image_shape).to(device)
    target_images_torch = target_images_torch.to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Optimization loop
    for epoch in range(num_epochs + 1):
        optimizer.zero_grad()
        output = model()  # .unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        # Compare with each image in the batch
        loss = criterion(output, target_images_torch)
        loss.mean().backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}')

    # Return the parameters
    return {'centers': model.centers.detach().cpu(),
            'covariances': model.covariances.detach().cpu(),
            'amplitudes': model.amplitudes.detach().cpu()}, loss

def optimize_rbf_iterative(target_images_np, max_num_rbf=20,
                           learning_rate=0.01, num_epochs=5000,
                           threshold=0.01):
    """
    For a given batch of images target_images_np, this function will 
    fit the parameters of max_num_rbf gaussian functions one at a time.
    """
    target_images = torch.tensor(target_images_np)
    done_indices = torch.tensor(0)
    done_centers = torch.empty(0, 0, 2)
    done_covariances = torch.empty(0, 0, 3)
    done_amplitudes = torch.empty(0, 0, 1)
    done_images = torch.empty(0, 0, target_images_np.shape[1])

    for i in range(max_num_rbf):
        print(f"Optimizing RBF {i + 1}/{max_num_rbf}")
        params, loss = optimize_rbf_coll(target_images_np, num_rbf=i + 1,
                                         learning_rate=learning_rate,
                                         num_epochs=num_epochs)
        # Check if the loss is below the threshold
        done_idx = torch.where(loss < threshold)[0]
        if done_idx.numel() > 0:
            done_indices = torch.cat((done_indices, done_idx), dim=0)
            for idx in done_idx:
                done_centers = torch.cat((done_centers,
                                         params['centers'][idx].unsqueeze(0)),
                                         dim=0)
                done_covariances = torch.cat((done_covariances,
                                              params['covariances'][idx].unsqueeze(0)),
                                              dim=0)
                done_amplitudes = torch.cat((done_amplitudes,
                                             params['amplitudes'][idx].unsqueeze(0)),
                                             dim=0)
                done_images = torch.cat((done_images,
                                         target_images[idx].unsqueeze(0)),
                                         dim=0)
            # and pop those indices from the 4 tensors
            centers = remove_idx(params['centers'], done_idx, dim=0)
            covariances = remove_idx(params['covariances'], done_idx, dim=0)
            amplitudes = remove_idx(params['amplitudes'], done_idx, dim=0)
            target_images = remove_idx(target_images, done_idx, dim=0)
    centers = torch.cat((done_centers, params['centers']), dim=0)
    covariances = torch.cat((done_covariances, params['covariances']), dim=0)
    amplitudes = torch.cat((done_amplitudes, params['amplitudes']), dim=0)
    images = torch.cat((done_images, target_images), dim=0)

    return (centers, covariances, amplitudes), images


def remove_idx(tensors, indices, dim):
    """
    Remove the indices from the tensors along the specified dimension.
    """
    if type(tensors) is torch.Tensor:
        out = torch.empty((0,) + tensors[0].shape)
        for i in indices:
            out = torch.cat((out, tensors[:i], tensors[i + 1:]), dim=dim)
    elif type(tensors) is np.ndarray:
        out = np.empty((0,) + tensors[0].shape)
        for i in indices:
            out = np.concatenate((out, tensors[:i], tensors[i + 1:]), axis=dim)
    else:
        raise TypeError("Unsupported type for tensors. Expected torch.Tensor or np.ndarray.")

    return out

def print_iamges(target_images_np, model):
    # Generate the replicated image (let's just take the first one for visualization)
    model.eval()
    with torch.no_grad():
        replicated_image_torch = model().cpu()
        replicated_image_np = replicated_image_torch.numpy()

    # Visualize the results (comparing the first target image with the generated one)
    for i in range(4):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(target_images_np[0])
        plt.title('First Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(replicated_image_np)
        plt.title('Generated Image (Shared Parameters)')
        plt.savefig(f'test{i}.png')


if __name__ == "__main__":
    debug = True
    path = "~/scratch/nanophoto/topoptim/fulloptim/images.npy"
    path = os.path.expanduser(path)
    num_images = -1 if not debug else 4
    num_epochs = 5000 if not debug else 100
    images = normalise(np.load(path))
    params, images = optimize_rbf_iterative(images, max_num_rbf=20,
                                                learning_rate=0.01,
                                            num_epochs=num_epochs,
                                                threshold=0.01) 
    coll = RBFImageCollection(
        num_images=images.shape[0], num_rbf=params[0].shape[1],
        image_shape=images.shape[2:], centers=params[0],
        covariances=params[1], amplitudes=params[2])
    print_iamges(images, coll)
    np.save(os.path.join(os.path.dirname(path), "gaussian_params.npy"),
            params.numpy())
