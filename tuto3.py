import gc
import os
import sys
import argparse
import yaml
from tqdm import trange

import numpy as np
import torch
# from torch.profiler import profile, record_function, ProfilerActivity

from tutoutils import (rbf, plot_samples_and_histogram, save_code,
                            CustomResNet152, tonumpy)
from nanophoto.models import ScatteringNetwork
from nanophoto.meep_compute_fom import compute_FOM_parallele
from nanophoto.get_trained_models import get_cpx_fields_unet_cnn_fompred
from set_transfo.models import SetTransformer

from icecream import ic, install
install()

sys.path.append(
    '/home/mila/l/letournv/repos/nanophoto/experiments/fom_predictors/')


def log_tensor_mem():
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


class PhotoEnv():
    def __init__(
        self,
        seq_len = 30,
        design_shape = (101,91),
        # action dim = 2 centers + 3 covariance matrix components + 1 amplitude
        action_dim=6,
        device_str="cuda",
        verify_actions=False,
        dtype=torch.float32,
    ):
        # Set verify_actions to False to disable action verification for faster step execution.

        state_shape=(seq_len, action_dim)
        self.state_shape = state_shape
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.device = torch.device(device_str)
        # self.terminal_action = torch.full(
        #     (self.action_dim,), -float("inf"), device=self.device)
        # self.sink_state = torch.full(
        #     self.state_shape, -float("inf"), device=self.device)
        self.verify_actions = verify_actions
        self.reward_fun = get_cpx_fields_unet_cnn_fompred()
        self.init_value = 0
        self.dtype = dtype

    def reward(self, x):
        images = rbf(x)
        images = torch.nn.functional.sigmoid(images)
        return self.reward_fun(images)

    def log_reward(self, x):
        # logre = torch.logsumexp(torch.stack([m.log_prob(x) for m in self.mixture], 0), 0)
        # x = (x.to(self.device) > 1/2).to(self.dtype)
        logre = torch.log(self.reward(x))
        return logre


def get_policy_dist(env, model, x, min_policy_std, max_policy_std):
    """
    A policy is a distribution we predict the parameters of using a neural
    network, which we then sample from.
    """
    x = x.contiguous()
    pf_params = model(x)  # Shape = [batch_shape, env.action_dim]
    pf_params = pf_params.squeeze()
    policy_mean = pf_params[:, 0: env.action_dim]
    policy_std = pf_params[:, env.action_dim:]
    policy_std = torch.clamp(policy_std, min_policy_std, max_policy_std)

    policy_dist = torch.distributions.Normal(policy_mean, policy_std)

    return policy_dist


def step(state: torch.Tensor, action: torch.Tensor):
    """
    Perform a single step in the environment. This is done by adding the
    action tensor to the state tensor at the smallest dim 1 index zero tensor.
    input: state: (batch, seq_len, action_dim)
    action: (batch, action_dim)
    output: new_state: (batch, seq_len, action_dim)
    """
    nonzero_mask = state[0].abs().sum(dim=1) > 0
    index = nonzero_mask.float().argmin()
    # get the first index of the first zero tensor
    new_state = torch.clone(state).detach()
    new_state[:, index, :] = action
    return new_state

def test__step():
    x = torch.zeros(2, 4, 2)
    z = torch.ones(2, 2, 2)
    x[:, :2, :] = z
    y = torch.rand(2, 2)
    xx = step(x, y)

    maskx = x.abs().sum(dim=2) > 0
    indx = maskx.float().argmin(dim=1)
    maskxx = xx.abs().sum(dim=2) > 0
    indxx = maskxx.float().argmin(dim=1)
    assert (indxx == indx + 1).all()


def initalize_state(batch_size, device, env, randn=False):
    """Trajectory starts at state = (X_0, t=0)."""
    shape = (batch_size,) + env.state_shape
    x = torch.zeros(shape, device=device)
    x[:, 0] = env.init_value

    return x


def setup_experiment(hid_dim=64, lr_model=1e-3, lr_logz=1e-1):
    """Generate the learned parameters and optimizer for an experiment.

    Forward and backward models are MLPs with a single hidden layer. logZ is
    a single parameter. Note that we give logZ a higher learning rate, which is
    a common trick used when utilizing Trajectory Balance.
    """
    # Input = [x_position, n_steps], Output = [mus, standard_deviations].
    dtype = torch.float
    device = torch.device('cuda')
    dim_input = env.action_dim
    dim_output = env.action_dim
    # breakpoint()
    forward_model = SetTransformer(dim_input=dim_input, num_outputs=1,
            dim_output=2*dim_output).to(device)
    backward_model = SetTransformer(dim_input=dim_input, num_outputs=1,
            dim_output=2*dim_output).to(device)
    ic("num model params", sum(param.numel() for param in forward_model.parameters()))

    logZ = torch.nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = torch.optim.Adam(
        [
            {'params': forward_model.parameters(), 'lr': lr_model},
            {'params': backward_model.parameters(), 'lr': lr_model},
            {'params': [logZ], 'lr': lr_logz},
        ]
    )

    return (forward_model, backward_model, logZ, optimizer)


def train(batch_size, trajectory_length, env, device, n_iterations,
        min_policy_std, max_policy_std, savepath):
    """Continuous GFlowNet training loop, with the Trajectory Balance objective."""
    # seed_all(seed)
    # Default hyperparameters used.
    forward_model, backward_model, logZ, optimizer = setup_experiment()
    losses = []
    tbar = trange(n_iterations)
    avg_log_reward = 0

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
            policy_dist = get_policy_dist(env, forward_model, x, min_policy_std, max_policy_std)
            action = policy_dist.sample()
            action.squeeze()
            logprob =policy_dist.log_prob(action)
            logprob.squeeze()
            logPF += logprob
            new_x = step(x, action)
            trajectory[:, t + 1, :] = action
            x = new_x

        del policy_dist, action, new_x
        torch.cuda.empty_cache()

        # Backward loop to compute logPB from existing trajectory under the backward policy.
        for t in range(trajectory_length, 0, -1):
            policy_dist = get_policy_dist(env, backward_model, x, min_policy_std, max_policy_std)
            action = trajectory[:, t, :] - trajectory[:, t - 1, :]
            logPB += policy_dist.log_prob(action)

        log_reward = env.log_reward(x)
        avg_log_reward = log_reward.mean()

        # Compute Trajectory Balance Loss.
        loss = (logZ + logPF - logPB - log_reward).pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if it % 100 == 0:
            tbar.set_description("Training iter {}: (loss={:.3f}, logZ={:.3f}, log reward={:.3f}".format(
                it,
                np.array(losses[-100:]).mean(),
                logZ.item(),
                avg_log_reward,
            )
            )
            os.makedirs('checkpoints/', exist_ok=True)
            torch.save(forward_model.state_dict(), 'checkpoints/forwardm.pth')
            torch.save(backward_model.state_dict(),
                       'checkpoints/backwardm.pth')
            torch.save(logZ, 'checkpoints/logz.pth')
            log_reward = tonumpy(log_reward.squeeze())
            x = tonumpy(torch.nn.functional.sigmoid(x))
            plot_samples_and_histogram(x, log_reward, 10, savepath)

    return (forward_model, backward_model, logZ)


def inference(trajectory_length, forward_model, env, batch_size, min_policy_std, max_policy_std):
    device = next(forward_model.parameters()).device
    with torch.no_grad():
        # trajectory = torch.zeros((batch_size, trajectory_length + 1, ), device=device)
        # trajectory[:, 0, 0] = env.init_value

        x = initalize_state(batch_size, device, env)

        for t in range(trajectory_length):
            policy_dist = get_policy_dist(env, forward_model, x, min_policy_std, max_policy_std)
            action = policy_dist.sample()

            x = step(x, action)
            # trajectory[:, t + 1, :] = new_x
            # x = new_x

    x = torch.nn.functional.sigmoid(x)
    return x


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store_true', default=False)
    args = parser.parse_args()
    debug = args.d

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    trajectory_length = 30
    min_policy_std = 0.1
    max_policy_std = 1.0
    batch_size = 16
    action_dim = 6
    scatter_config = {"scattering_scale": 2, "scattering_angles": 8,
                      "scattering_max_order": 2, "mlp_dim": 1024, "num_layers": 1,
                      "output_dim": action_dim*2}

    savedir = os.environ['SLURM_JOB_ID'] if debug is False else 'debug'
    savepath = os.path.join('outfiles', savedir)
    os.makedirs(savepath, exist_ok=True)
    with open(os.path.join(savepath, 'config.yml'), 'w') as f:
        yaml.dump(scatter_config, f)
    save_code(savepath)

    n_iterations = 5000 if debug is False else 2
    reward_fn = get_cpx_fields_unet_cnn_fompred()
    env = PhotoEnv()
    # with profile(activities=[ProfilerActivity.CUDA], profile_memory=True) as prof:
    forward_model, backward_model, logZ = train(
    batch_size, trajectory_length, env, device, n_iterations,
    min_policy_std, max_policy_std, savepath)
    # with open("cuda_prof.log", "w") as f:
    #     old_sdtout = sys.stdout
    #     sys.stdout = f
    #     print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    #     sys.stdout = old_sdtout

    samples = inference(trajectory_length, forward_model, env, batch_size,
            min_policy_std, max_policy_std)
    samples = samples.cpu().numpy()
    if debug:
        fom = torch.rand(samples.shape[0])
    else:
        fom = compute_FOM_parallele(samples)
    np.save(os.path.join(savepath, 'samples.npy'), samples)
    np.save(os.path.join(savepath, 'fom.npy'), fom)
    plot_samples_and_histogram(samples, fom, 10, savepath)
    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            # print(prof.key_averages().table(sort_by="cuda_memory_usage"))
