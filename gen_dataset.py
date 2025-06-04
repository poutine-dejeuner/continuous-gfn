import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

def normalise(image):
    return (image - image.min()) / (image.max() - image.min())

def rbf(centers, covariances, amplitudes, grid_points, image_shape):
    assert centers.shape[0:2] == covariances.shape[0:2] == amplitudes.shape[0:2]

    device = centers.device
    dtype = centers.dtype

    cov_shape = covariances.shape[:-1] +  (2, 2)
    covariance_mat = torch.zeros(size=cov_shape, device=device)
    covariance_mat[..., 0, 0] = covariances[..., 0]
    covariance_mat[..., 1, 1] = covariances[..., 1]
    covariance_mat[..., 0, 1] = covariances[..., 2]
    covariance_mat[..., 1, 0] = covariances[..., 2]

    centers = centers.unsqueeze(-2)
    n_gridpoints = grid_points.shape[0]
    centers = torch.repeat_interleave(centers, repeats=n_gridpoints, dim=-2)
    diff = grid_points - centers
    z = torch.einsum('...ij,...jk,...ik->...i', diff, covariance_mat, diff)
    z = torch.exp(-0.5 * z)
    amplitudes = amplitudes.expand_as(z)
    z = z * amplitudes
    z = z.sum(-2)
    z_shape = z.shape[:-1] + (image_shape[0], image_shape[1])
    z = z.reshape(z_shape)
    # z = torch.transpose(z, -1, -2)
    assert z.shape == (centers.shape[0], image_shape[0], image_shape[1])
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
            self.dtype = centers.dtype
            assert centers.shape[0] == covariances.shape[0] == amplitudes.shape[0], \
                f"Centers shape: {centers.shape}, " \
                f"Covariances shape: {covariances.shape}, " \
                f"Amplitudes shape: {amplitudes.shape}"

            assert centers.ndim == covariances.ndim == amplitudes.ndim ==  3, \
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
            self.dtype = torch.float32
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

        self.grid = make_grid(rows, cols).to(self.device, self.dtype)

    def rbf_function(self, coords):
        # x, y = coords[:,0], coords[:,1]
        out = rbf(self.centers, self.covariances, self.amplitudes, coords,
                  self.image_shape)
        out = out.unsqueeze(1)
        return out


    def forward(self):
        output = self.rbf_function(self.grid)
        return torch.sigmoid(output)


def make_grid(rows, cols):
    """
    Makes a grid of (row * cols) points in the range [-1, 1]. returns a tensor
    of shape (rows * cols, 2) where each row is a point (x, y) in the grid.
    """
    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack((X.flatten(), Y.flatten()), axis=-1)
    grid_points = torch.tensor(grid_points)

    return grid_points

def optimize_rbf_coll(target_images_np, parameters=torch.empty(0), num_rbf=20, learning_rate=0.01, num_epochs=5000):
    assert target_images_np.min() >= 0 and target_images_np.max() <= 1

    if target_images_np.ndim == 2:
        image_shape = target_images_np.shape
    elif target_images_np.ndim == 3:
        image_shape = target_images_np.shape[1:]
    else:
        raise ValueError("target_images_np must be 2D or 3D array.")
    num_images = target_images_np.shape[0]
    target_images_torch = torch.tensor(np.array(
        target_images_np), dtype=torch.float32).unsqueeze(1)  # [batch_size, C, H, W]

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RBFImageCollection(num_images, num_rbf, image_shape, *parameters).to(device)
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
        loss = loss.mean(dim=tuple(range(1, loss.ndim)))
        meanloss = loss.mean()
        meanloss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f'Epoch {epoch}/{num_epochs}, Loss: {meanloss.item():.4f}')

    # Return the parameters
    return {'centers': model.centers.detach(),
            'covariances': model.covariances.detach(),
            'amplitudes': model.amplitudes.detach()}, loss, model

def optimize_rbf_iterative_oneimage(target_image_np, max_num_rbf=20,
                           learning_rate=0.01, num_epochs=5000,
                           threshold=0.01):
    centers = torch.empty(1, 0, 2)
    covariances = torch.empty(1, 0, 3)
    amplitudes = torch.empty(1, 0, 1)

    for i in range(max_num_rbf):
        params = [centers, covariances, amplitudes]
        print(f"Optimizing RBF {i + 1}/{max_num_rbf}")
        new_params, loss = optimize_rbf_coll(params, target_image_np, num_rbf=i + 1,
                                         learning_rate=learning_rate,
                                         num_epochs=num_epochs)
        if loss < threshold:
            break
    centers = torch.cat((centers, new_params['centers']), dim=1)
    covariances = torch.cat((covariances, new_params['covariances']), dim=1)
    amplitudes = torch.cat((amplitudes, new_params['amplitudes']), dim=1)

    return (centers, covariances, amplitudes)

def optimize_rbf_iterative(target_images_np, max_num_rbf=20,
                           learning_rate=0.01, num_epochs=5000,
                           threshold=0.01):
    """
    For a given batch of images target_images_np, this function will 
    fit the parameters of max_num_rbf gaussian functions one at a time.
    """
    target_images = torch.tensor(target_images_np)
    done_indices = torch.empty(0)
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

def print_images(target_images_np, model):
    savepath = 'test_img'
    os.makedirs(savepath, exist_ok=True)
    # Generate the replicated image (let's just take the first one for visualization)
    model.eval()
    with torch.no_grad():
        replicated_image_torch = model().cpu()
        replicated_image_np = replicated_image_torch.numpy().squeeze()

    # Visualize the results (comparing the first target image with the generated one)
    for i in range(4):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(target_images_np[0].squeeze())
        plt.title('First Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(replicated_image_np[i])
        plt.title('Generated Image (Shared Parameters)')
        plt.savefig(os.path.join(savepath, f'test{i}.png'))


def eval_single_param_error(target_images, params, grid, image_shape):
    """
    Prend des paramètres de RBF params fittés aux images target_images et prend
    chaque slice de param pour générer une image utilisant un seul triplet
    centre, covariance, amplitude et calcule l'erreur avec l'image cible. Les
    paramètres sont classés selon cette erreure.
    """
    if type(target_images) is np.ndarray:
        target_images = torch.tensor(target_images, dtype=torch.float32,
                                     device=params[0].device)
    centers, covariances, amplitudes = params
    assert centers.shape[0] == covariances.shape[0] == amplitudes.shape[0], \
        f"Centers shape: {centers.shape}, " \
        f"Covariances shape: {covariances.shape}, " \
        f"Amplitudes shape: {amplitudes.shape}"

    errors = torch.empty((centers.shape[0], 0), device=centers.device)
    for i in range(centers.shape[1]):
        image = rbf(centers[:,i:i+1,:], covariances[:,i:i+1,:],
                    amplitudes[:,i:i+1,:],
                    grid, image_shape)
        error = torch.abs(image - target_images).mean(dim=(-2, -1))
        errors = torch.cat((errors, error.unsqueeze(1)), dim=1)
    assert errors.shape == centers.shape[:-1], f"""{errors.shape},
                                                {centers.shape}"""
    return errors


def retire_rbf_params_faibles(images, params, grid, threshold=0.01):
    device = params[0].device
    errors = eval_single_param_error(images, params, grid, images.shape[-2:])
    errors, idx = torch.sort(errors, dim=1, descending=True)
    params = torch.cat(params, dim=-1)
    # reorder params in dim=1 according to the sorted indices
    params = torch.gather(params, dim=1, index=idx.unsqueeze(-1).expand(
        -1, -1, params.shape[-1]))

    # get smallest column index where the mask is True, if none then set to -1
    mask = errors > threshold
    idx = torch.where(mask.any(dim=1), mask.float().argmax(dim=1),
                      torch.full((errors.size(0),), -1).to(device))
    # make a mask that is True for indices less than idx
    row_idx = torch.arange(errors.size(1)).unsqueeze(0).expand(errors.size(0),
                                                               -1).to(device)
    mask = (row_idx <= idx.unsqueeze(1)).int()

    # for each element in params, set entries to 0 for indices greater than idx
    mask = mask.unsqueeze(-1).expand_as(params)
    params = params * mask
    return params

def fit_une_image():
    # debug = True
    debug = False
    path = "~/scratch/nanophoto/topoptim/fulloptim/images.npy"
    path = os.path.expanduser(path)
    num_epochs = 5000 if not debug else 100
    max_num_rbf = 30 if not debug else 2
    image = normalise(np.load(path)[:1])
    params = optimize_rbf_iterative_oneimage(image, max_num_rbf=max_num_rbf,
                                                learning_rate=0.01,
                                            num_epochs=num_epochs,
                                                threshold=0.01) 

    coll = RBFImageCollection(
        num_images=image.shape[0], num_rbf=params[0].shape[1],
        image_shape=image.shape[-2:], centers=params[0],
        covariances=params[1], amplitudes=params[2])
    print_images(image, coll)
    params = torch.cat(params, dim=-1)
    np.save(os.path.join(os.path.dirname(path), "gaussian_params.npy"),
            params.numpy())

def test__rbf():
    n_im = 4
    param = torch.rand(n_im, 20, 6)
    grid = make_grid(101, 91).to(torch.float)
    im = rbf(param[:, :, :2], param[:, :, 2:5], param[:, :, 5:], grid, (101,
             91))
    _, axes = plt.subplots(n_im, 1, figsize=(10, 10))
    for i in range(n_im):
        axes[i].imshow(im[i].cpu().numpy())
        axes[i].set_axis_off()
        # axes[i].set_title(f"Image {i + 1}")
    plt.show()

def fit_topoptim():
    debug = False
    # debug = True
    path = "~/scratch/nanophoto/topoptim/fulloptim/images.npy"
    path = os.path.expanduser(path)
    # num_images = -1 if not debug else 4
    num_images = 4
    num_epochs = 2000 if not debug else 100
    num_rbf = 50 if not debug else 2
    learning_rate = 0.1
    threshold = 0.01

    images = normalise(np.load(path)[:num_images])

    params, loss, model = optimize_rbf_coll(images, torch.empty(0), num_rbf=num_rbf,
                                                learning_rate=learning_rate,
                                            num_epochs=num_epochs) 

    params = (params['centers'], params['covariances'], params['amplitudes'])
    grid = model.grid
    params = retire_rbf_params_faibles(images, params, grid, threshold=threshold)
    centers = params[...,:2]
    covariances = params[...,2:5]
    amplitudes = params[...,5:]

    coll = RBFImageCollection(
        num_images=images.shape[0], num_rbf=params[0].shape[1],
        image_shape=images.shape[-2:], centers=centers,
        covariances=covariances, amplitudes=amplitudes)

    print_images(images, coll)
    np.save(os.path.join(os.path.dirname(path), "gaussian_params.npy"),
            params.cpu().numpy())


if __name__ == "__main__":
    # test__rbf()
    fit_topoptim()
