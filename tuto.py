import sys
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange
from pynvml import *
import torch.autograd.profiler as profiler

from tutoutils import rbf, plot_samples_and_histogram, get_vram
import nanophoto.models as models
from nanophoto.models import ScatteringNetwork, convnet
from nanophoto.meep_compute_fom import compute_FOM_parallele
from nanophoto.get_trained_models import get_cpx_fields_unet_cnn_fompred

from icecream import ic

sys.path.append('/home/mila/l/letournv/repos/nanophoto/experiments/fom_predictors/')

import gc

def log_tensor_mem():
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                mem = obj.element_size() * obj.nelement()
                print(f"{type(obj)} | shape: {tuple(obj.shape)} | {mem / 1024**2:.2f} MB")
                total += mem
        except Exception:
            pass
    print(f"[Total Tensor GPU Mem] {total / 1024**2:.2f} MB\n")

class PhotoEnv():
    def __init__(
        self,
        state_shape=(101, 91),
        # action dim = 2 centers + 3 covariance matrix components + 1 amplitude
        action_dim=6,
        device_str="cuda",
        verify_actions=False,
        dtype=torch.float32,
    ):
        # Set verify_actions to False to disable action verification for faster step execution.

        self.state_shape = state_shape
        self.action_dim = action_dim
        self.device = torch.device(device_str)
        self.terminal_action = torch.full(
            (self.action_dim,), -float("inf"), device=self.device)
        self.sink_state = torch.full(
            self.state_shape, -float("inf"), device=self.device)
        self.verify_actions = verify_actions
        self.reward_fun = get_cpx_fields_unet_cnn_fompred()
        self.init_value = 0
        self.dtype = dtype

    def reward(self, x):
        return self.reward_fun(x)

    def log_reward(self, x):
        # logre = torch.logsumexp(torch.stack([m.log_prob(x) for m in self.mixture], 0), 0)
        # x = (x.to(self.device) > 1/2).to(self.dtype)
        logre = torch.log(self.reward_fun(x))
        return logre


def get_policy_dist(model, x):
    """
    A policy is a distribution we predict the parameters of using a neural
    network, which we then sample from.
    """
    x = x.contiguous()
    pf_params = model(x)  # Shape = [batch_shape, env.action_dim]
    policy_mean = pf_params[:, 0: env.action_dim]
    policy_std = pf_params[:, env.action_dim:]
    policy_std = torch.clamp(policy_std, min_policy_std, max_policy_std)

    policy_dist = torch.distributions.Normal(policy_mean, policy_std)

    return policy_dist

def action_parameters_to_state_space(action):
    action_image = rbf(action, env.state_shape)
    return action_image

def step(x, action):
    """Takes a forward step in the environment."""
    new_x = torch.zeros_like(x)
    action = action_parameters_to_state_space(action)
    new_x = x + action

    return new_x


def initalize_state(batch_size, device, env, randn=False):
    """Trajectory starts at state = (X_0, t=0)."""
    x = torch.zeros((batch_size,) + env.state_shape, device=device)
    x[:, 0] = env.init_value

    return x

def setup_experiment(hid_dim=64, lr_model=1e-3, lr_logz=1e-1):
    """Generate the learned parameters and optimizer for an experiment.

    Forward and backward models are MLPs with a single hidden layer. logZ is
    a single parameter. Note that we give logZ a higher learning rate, which is
    a common trick used when utilizing Trajectory Balance.
    """
    # Input = [x_position, n_steps], Output = [mus, standard_deviations].
    forward_model = ScatteringNetwork(**scatter_config)
    backward_model = ScatteringNetwork(**scatter_config)

    logZ = torch.nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = torch.optim.Adam(
        [
            {'params': forward_model.parameters(), 'lr': lr_model},
            {'params': backward_model.parameters(), 'lr': lr_model},
            {'params': [logZ], 'lr': lr_logz},
        ]
    )

    return (forward_model, backward_model, logZ, optimizer)

def train(batch_size, trajectory_length, env, device, n_iterations):
    """Continuous GFlowNet training loop, with the Trajectory Balance objective."""
    # seed_all(seed)
    forward_model, backward_model, logZ, optimizer = setup_experiment()  # Default hyperparameters used.
    losses = []
    tbar = trange(n_iterations)

    for it in tbar:
        optimizer.zero_grad()

        x = initalize_state(batch_size, device, env)

        # Trajectory stores all of the states in the forward loop.
        traj_shape = (batch_size, trajectory_length + 1, env.action_dim)
        trajectory = torch.zeros(traj_shape, device=device)
        logPF = torch.zeros((batch_size, env.action_dim), device=device)
        logPB = torch.zeros((batch_size, env.action_dim), device=device)

        # Forward loop to generate full trajectory and compute logPF.
        for t in range(trajectory_length):
            policy_dist = get_policy_dist(forward_model, x)
            action = policy_dist.sample()
            logPF += policy_dist.log_prob(action)

            new_x = step(x, action)
            trajectory[:, t + 1, :] = action
            x = new_x

        del policy_dist, action, new_x
        torch.cuda.empty_cache()

        # Backward loop to compute logPB from existing trajectory under the backward policy.
        for t in range(trajectory_length, 0, -1):
            policy_dist = get_policy_dist(backward_model, x)
            action = trajectory[:, t, :] - trajectory[:, t - 1, :]
            logPB += policy_dist.log_prob(action)

        log_reward = env.log_reward(x)

        # Compute Trajectory Balance Loss.
        loss = (logZ + logPF - logPB - log_reward).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if it % 100 == 0:
            tbar.set_description("Training iter {}: (loss={:.3f}, estimated logZ={:.3f}, LR={}".format(
                it,
                np.array(losses[-100:]).mean(),
                logZ.item(),
                optimizer.param_groups[0]['lr'],
                )
            )

    return (forward_model, backward_model, logZ)

def inference(trajectory_length, forward_model, env, batch_size=1000):
    with torch.no_grad():
        # trajectory = torch.zeros((batch_size, trajectory_length + 1, ), device=device)
        # trajectory[:, 0, 0] = env.init_value

        x = initalize_state(batch_size, device, env)

        for t in range(trajectory_length):
            policy_dist = get_policy_dist(forward_model, x)
            action = policy_dist.sample()

            x = step(x, action)
            # trajectory[:, t + 1, :] = new_x
            # x = new_x

    return x

def plot_samples_and_histogram(samples, scores,N):
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
                axs[i, j].imshow(samples[index], cmap='gray')
                axs[i, j].set_title(f'Score: {scores[index]:.2f}')
                axs[i, j].axis('off')
            else:
                axs[i, j].axis('off')

    # Save the samples figure
    plt.savefig('samples.png')
    plt.close()

    # Create a histogram of scores
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')

    # Save the histogram figure
    plt.savefig('histogram.png')
    plt.close()


debug = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

trajectory_length = 20
min_policy_std = 0.1
max_policy_std = 1.0
batch_size = 64
action_dim = 6
scatter_config={"scattering_scale":2, "scattering_angles":8,
        "scattering_max_order":2, "mlp_dim":1024, "num_layers":1,
                "output_dim":action_dim*2, "device":device, "dtype":dtype}
n_iterations = 5000
reward_fn = get_cpx_fields_unet_cnn_fompred()
env = PhotoEnv()
forward_model, backward_model, logZ = train(batch_size, trajectory_length, env, device, n_iterations)

samples = inference(trajectory_length, forward_model, env)
samples = samples.cpu().numpy()
if debug:
    fom = torch.rand(samples.shape[0])
else:
    fom = compute_FOM_parallele(samples)
plot_samples_and_histogram(samples, fom, 10)
