import os, sys
from functools import wraps
import linecache
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

class CustomResNet152(nn.Module):
    def __init__(self, input_channels=1, out_features=12):
        super().__init__()
        self.base = models.resnet152()
        self.base.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base.fc = nn.Linear(self.base.fc.in_features, out_features)

    def forward(self, x):
        return self.base(x)

def trace_gpu_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        mainfile = os.path.abspath(func.__code__.co_filename)

        def tracer(frame, event, arg):
            if event == 'line':
                filename = os.path.abspath(frame.f_code.co_filename)
                if filename != mainfile:
                    return tracer  # skip other files

                lineno = frame.f_lineno
                line = linecache.getline(filename, lineno).strip()
                print(f"[line {lineno}] {line}")
                ic(torch.cuda.memory_allocated()/1e9)

            return tracer

        sys.settrace(tracer)
        try:
            return func(*args, **kwargs)
        finally:
            sys.settrace(None)

    return wrapper

def model_mem_use(model):
    """
    Get the memory usage of a model.
    out: number of parameters, memory use
    """
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    mem = num_params * param.element_size()
    return num_params, mem / (1024 ** 2)  # Convert to MB

def get_model_grad_norm(model):
    norm = 0
    for p in model.parameters():
        norm += torch.norm(p.grad.flatten())**2
    norm = torch.sqrt(norm).item()
    return norm

def save_code(savepath):
    if not os.path.exists(os.path.join(savepath, 'code.py')):
        sourcefile = os.path.basename(__file__)
        shutil.copy2(sourcefile, os.path.join(savepath, 'code.py'))

def get_vram():
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    total = torch.cuda.mem_get_info()[1] / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return f'VRAM: {total - free:.2f}/{total:.2f}GB\t VRAM:[' + (
            total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'

def get_ram():
    import psutil
    mem = psutil.virtual_memory()
    free = mem.available / 1024 ** 3
    total = mem.total / 1024 ** 3
    total_cubes = 24
    free_cubes = int(total_cubes * free / total)
    return f'RAM: {total - free:.2f}/{total:.2f}GB\t RAM:[' + (total_cubes - free_cubes) * '▮' + free_cubes * '▯' + ']'

def tonumpy(tensor):
    return tensor.detach().cpu().numpy()

def stats(array):
    ic(array.min(), array.max(), array.mean())

def normalise(images):
    if (images == 0.).all() or (images == 1.).all():
        return images
    return (images - images.min()) / (images.max() - images.min())

def plot_samples_and_histogram(samples, scores, N, savepath):
    """
        makes a NxN figure of samples with their score values and a
        histogram of the scores. saves the figures to 2 png files.
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(N, N, figsize=(10, 10))
    fig.suptitle('Samples and Scores')

    # Plot samples in the grid
    for i in range(N):
        for j in range(N):
            index = i * N + j
            if index < len(samples):
                axs[i, j].imshow(samples[index])
                axs[i, j].set_title(f'Score: {scores[index]:.2f}')
                axs[i, j].axis('off')
            else:
                axs[i, j].axis('off')

    # Save the samples figure
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'samples.png'))
    plt.close()

    # Create a histogram of scores
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    # Save the histogram figure
    plt.savefig(os.path.join(savepath, 'histogram.png'))
    plt.clf()

def rbf(parameters, image_shape=(101,91)):                                              
    """                                                                        
        takes a parameters tensor of shape (M, N, 6), in dim -1, the first 2       
        components are the center of the gaussian, the next 3 are the          
        components of the covariance matrix, and the last one is the amplitude.
        it computes the gaussian function for each of the N samples at every   
        points of a grid of shape image_shape.                                 
        input:
        parameters: torch.Tensor of shape (..., N, 6)                               
        image_shape: tuple (height, width)                                     

        returns:
        image: torch.Tensor (..., height, width)

    """
    dtype = torch.float
    paramtype = type(parameters)
    device = parameters.device
    parameters = parameters.to(device)

    # Unpack parameters
    centers = parameters[..., :, :2]
    centers[...,0] = centers[...,0]
    centers[...,1] = centers[...,1]
    covariances = parameters[..., :, 2:5]
    amplitudes = parameters[..., :, 5]
    # covariance_mat is an array of shape (N, 2, 2)
    covariance_mat = torch.zeros(parameters.shape[:-1] +  (2, 2), device=device)
    covariance_mat[..., 0, 0] = covariances[..., 0]
    covariance_mat[..., 1, 1] = covariances[..., 1]
    covariance_mat[..., 0, 1] = covariances[..., 2]
    covariance_mat[..., 1, 0] = covariances[..., 2]
    assert centers[...,0].min() >= -1
    assert centers[...,1].min() >= -1
    assert centers[...,0].max() <= 1
    assert centers[...,1].max() <= 1


    # Create a grid of points
    x = np.linspace(-1, 1, image_shape[0])
    y = np.linspace(-1, 1, image_shape[1])
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack((X.flatten(), Y.flatten()), axis=-1)

    n_gridpoints = grid_points.shape[0]
    # grid_points = np.expand_dims(grid_points, axis=0)
    # grid_points = np.repeat(grid_points, len(parameters), axis=0)
    grid_points = torch.tensor(grid_points, dtype=dtype, device=device)
    
    # grid_points is an array of shape (N, height * width, 2)
    # Compute the difference from the centers
    # centers = centers.unsqueeze(1)
    # centers = centers.repeat(1, num_gridpoints, 1)
    centers = centers.unsqueeze(-2)
    centers = torch.repeat_interleave(centers, repeats=n_gridpoints, dim=-2)
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

class RBFImageCollection(nn.Module):
    def __init__(self, num_images, num_rbf, image_shape, weights=None,
                    centers=None, widths=None, bias=None):
        super().__init__()
        self.num_rbf = num_rbf
        self.image_shape = image_shape
        rows, cols = image_shape
        # init weights, centers, widths and bias to random values if they are
        # not provided
        if (weights is not None and centers is not None and widths is not None
            and bias is not None):
            self.weights = nn.Parameter(weights)
            self.centers = nn.Parameter(centers)
            self.log_widths = nn.Parameter(widths)
            self.bias = nn.Parameter(bias)
        else:
            self.weights = nn.Parameter(torch.randn(num_rbf, num_images))
            self.centers = nn.Parameter(torch.rand(
                num_rbf, 2) * torch.tensor([cols, rows], dtype=torch.float32))
            self.log_widths = nn.Parameter(torch.randn(num_rbf, num_images))
            self.bias = nn.Parameter(torch.randn(1) * 0.1)

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

    def forward(self):
        device = self.weights.device
        rows, cols = self.image_shape
        r = torch.arange(rows, dtype=torch.float32)
        c = torch.arange(cols, dtype=torch.float32)
        r_grid, c_grid = torch.meshgrid(r, c, indexing='ij')
        coords = (c_grid.to(device), r_grid.to(device))
        output = self.rbf_function(coords)
        return torch.sigmoid(output)

def log_tensor_mem():
    import gc
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                mem = obj.element_size() * obj.nelement()
                print(
                    f"{type(obj)} | shape: {tuple(obj.shape)} | {mem / 1024**2:.2f} MB")
                total += mem
        except Exception:
            pass
    print(f"[Total Tensor GPU Mem] {total / 1024**2:.2f} MB\n")


if __name__ == "__main__":
    def test__plot_samples_and_histogram():
        """
            test the plot_samples_and_histogram function
        """
        # Create a grid of samples and scores
        N = 5
        num_samples = N * N
        samples = [np.random.rand(28, 28) for _ in range(num_samples)]
        scores = np.random.rand(num_samples)

        # Call the function to plot samples and histogram
        plot_samples_and_histogram(samples, scores, N)
        print('ok')

    def test__rbf():
        """
            test the rbf function
        """
        # Create random parameters
        device = torch.device('cuda')
        N = 10
        parameters = torch.rand(N, 6, device=device)
        image_shape = (28, 28)

        # Call the function to compute RBF
        results = rbf(parameters, image_shape)
        results = results.detach().cpu().numpy() 
        results = results > 1/2

        # Check the shape of the results
        assert results.shape == (N, image_shape[0], image_shape[1]), "Shape mismatch"
        print('ok')
        plot_samples_and_histogram(results, np.random.rand(N), 4)

    # Run the tests
    test__plot_samples_and_histogram()
    
    test__rbf()

