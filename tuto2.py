"""
    state space (101, 91)
    action space (6,)
"""
import os
import sys
import argparse
from tqdm import trange

import numpy as np
import torch

from tutoutils import (rbf, plot_samples_and_histogram, save_code,
                       CustomResNet152, stats)
from nanophoto.meep_compute_fom import compute_FOM_parallele
from nanophoto.get_trained_models import get_cpx_fields_unet_cnn_fompred

from orion.client import report_objective
from icecream import ic

from torch.profiler import profile, record_function, ProfilerActivity

sys.path.append(
    '/home/mila/l/letournv/repos/nanophoto/experiments/fom_predictors/')


"""
Le modele genere un seul design. Il y a un probleme avec les modeles PF et
PB...
TODO: il faut qu'a l'initialisation les forward/backward models produisent des
valeurs qui correspondent a l'espace d'action.
"""


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
        self.reward_fun = get_cpx_fields_unet_cnn_fompred().eval()
        # ic(model_mem_use(self.reward_fun))

        self.init_value = 1/2 
        self.dtype = dtype

    def reward_with_sigmoid(self, x):
        x = torch.nn.functional.sigmoid(x)
        return self.reward_fun(x)
    
    def reward(self, x):
        return self.reward_fun(x)

    def log_reward(self, x):
        # logre = torch.logsumexp(torch.stack([m.log_prob(x) for m in self.mixture], 0), 0)
        # x = (x.to(self.device) > 1/2).to(self.dtype)
        # m0 = torch.cuda.memory_allocated()
        logre = self.reward(x)
        # m1 = torch.cuda.memory_allocated()
        # print("reward comp mem use")
        # ic((m1-m0)/1024**2)
        logre = torch.log(logre)
        return logre


def get_policy_dist(env, model, x, min_policy_std, max_policy_std):
    """
    A policy is a distribution we predict the parameters of using a neural
    network, which we then sample from.
    """
    x = x.unsqueeze(1)
    x = x.contiguous()
    pf_params = model(x)  # Shape = [batch_shape, env.action_dim]
    policy_mean = pf_params[:, 0: env.action_dim]
    policy_std = pf_params[:, env.action_dim:]
    policy_std = torch.clamp(policy_std, min_policy_std, max_policy_std)
    policy_dist = torch.distributions.Normal(policy_mean, policy_std)
    return policy_dist


def action_parameters_to_state_space(action, state_shape=(101, 91)):
    action_image = rbf(action, state_shape)
    return action_image


def step(x, action):
    """Takes a forward step in the environment."""
    new_x = torch.zeros_like(x)
    action = action_parameters_to_state_space(action).to(x.device)
    new_x = x + action
    if (new_x > 1.).any() or (new_x < 0).any():
        print(new_x.min(), new_x.max())
        new_x = torch.clip(new_x, min=0, max=1)
        print(new_x.min(), new_x.max())
    return new_x

def step_with_sigmoid(x, action):
    new_x = torch.zeros_like(x)
    action = action_parameters_to_state_space(action).to(x.device)
    new_x = x + action
    new_x = torch.sigmoid(new_x)
    return new_x


def initalize_state(batch_size, device, env, randn=False):
    """Trajectory starts at state = (X_0, t=0)."""
    x = torch.zeros((batch_size,) + env.state_shape, device=device)
    x[:, 0] = env.init_value
    return x


def setup_experiment(lr_model=1e-3, lr_logz=1e-1, **kwargs):
    """Generate the learned parameters and optimizer for an experiment.

    Forward and backward models are MLPs with a single hidden layer. logZ is
    a single parameter. Note that we give logZ a higher learning rate, which is
    a common trick used when utilizing Trajectory Balance.
    """
    # Input = [x_position, n_steps], Output = [mus, standard_deviations].
    dtype = torch.float
    device = torch.device('cuda')
    forward_model = CustomResNet152(out_features=12).to(device)
    backward_model = CustomResNet152(out_features=12).to(device)

    logZ = torch.nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = torch.optim.Adam(
        [
            {'params': forward_model.parameters(), 'lr': lr_model},
            {'params': backward_model.parameters(), 'lr': lr_model},
            {'params': [logZ], 'lr': lr_logz},
        ]
    )
    return (forward_model, backward_model, logZ, optimizer)

# @trace_gpu_memory


def train(batch_size, trajectory_length, env, device, n_iterations,
          savepath, **setup_configs):
    """Continuous GFlowNet training loop, with the Trajectory Balance objective."""
    # seed_all(seed)
    # Default hyperparameters used.
    forward_model, backward_model, logZ, optimizer = setup_experiment(
        **setup_configs)
    min_policy_std = setup_configs['min_policy_std']
    max_policy_std = setup_configs['max_policy_std']
    losses = []
    tbar = trange(n_iterations)
    avg_log_reward = 0

    for it in tbar:
        # ic(torch.cuda.memory_allocated()/1e9)
        optimizer.zero_grad()

        x = initalize_state(batch_size, device, env)

        # Trajectory stores all of the states in the forward loop.
        traj_shape = (batch_size, trajectory_length + 1, env.action_dim)
        trajectory = torch.zeros(traj_shape, device=device)
        logPF = torch.zeros((batch_size, env.action_dim), device=device)
        logPB = torch.zeros((batch_size, env.action_dim), device=device)

        # Forward loop to generate full trajectory and compute logPF.
        for t in range(trajectory_length):
            policy_dist = get_policy_dist(
                env, forward_model, x, min_policy_std, max_policy_std)
            action = policy_dist.sample()
            logPF += policy_dist.log_prob(action)

            new_x = step_with_sigmoid(x, action)
            trajectory[:, t + 1, :] = action
            x = new_x

        del policy_dist, action, new_x
        torch.cuda.empty_cache()

        # Backward loop to compute logPB from existing trajectory under the backward policy.
        for t in range(trajectory_length, 0, -1):
            policy_dist = get_policy_dist(
                env, backward_model, x, min_policy_std, max_policy_std)
            action = trajectory[:, t, :] - trajectory[:, t - 1, :]
            logPB += policy_dist.log_prob(action)

        log_reward = env.log_reward(x)
        avg_log_reward = log_reward.mean().item()

        # Compute Trajectory Balance Loss.
        loss = (logZ + logPF - logPB - log_reward).pow(2).mean()
        print('backward mem use')
        # m0 = torch.cuda.memory_allocated()
        loss.backward()
        # m1 = torch.cuda.memory_allocated()
        # ic((m1-m0)/1024**2)
        # ic(m1/1024**2)

        # ic(get_model_grad_norm(forward_model))
        # ic(get_model_grad_norm(backward_model))

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
            torch.save(forward_model.state_dict(),
                       os.path.join(savepath, 'forwardm.pth'))
            torch.save(backward_model.state_dict(),
                       os.path.join(savepath, 'backwardm.pth'))
            torch.save(logZ, os.path.join(savepath, 'logz.pth'))
            # log_reward = tonumpy(log_reward.squeeze())
            # x = tonumpy(torch.nn.functional.sigmoid(x))
            # plot_samples_and_histogram(x, log_reward, 10, savepath)
        del x, logPF, logPB, log_reward
        torch.cuda.empty_cache()

    ic(torch.cuda.max_memory_allocated()/1024**2)
    return (forward_model, backward_model, logZ)


def inference(trajectory_length, forward_model, env, batch_size,
              min_policy_std, max_policy_std, trajectory_len):
    torch.cuda.reset_peak_memory_stats()
    device = next(forward_model.parameters()).device
    with torch.no_grad():
        # trajectory = torch.zeros(
        #     (batch_size, trajectory_length + 1, ), device=device)
        # trajectory[:, 0, 0] = env.init_value

        x = initalize_state(batch_size, device, env)

        for t in range(trajectory_len):
            policy_dist = get_policy_dist(
                env, forward_model, x, min_policy_std, max_policy_std)
            action = policy_dist.sample()

            x = step(x, action)
            # trajectory[:, t + 1, :] = new_x
            # x = new_x

    x = torch.nn.functional.sigmoid(x)
    print('size of largest tensor mem use')
    ic(torch.cuda.max_memory_allocated()/1024**2)
    return x

# @profile


def main(debug=True, **setup_configs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    trajectory_length = 32
    min_policy_std = setup_configs['min_policy_std']
    max_policy_std = setup_configs['max_policy_std']
    batch_size = 8
    action_dim = 6
    n_iterations = 5000 if debug is False else 2

    savedir = os.environ['SLURM_JOB_ID'] if debug is False else 'debug'
    savepath = os.path.join('outfiles', savedir)
    os.makedirs(savepath, exist_ok=True)
    save_code(savepath)

    reward_fn = get_cpx_fields_unet_cnn_fompred()
    env = PhotoEnv()
    forward_model, backward_model, _ = train(
        batch_size, trajectory_length, env, device, n_iterations,
        savepath, **setup_configs)

    inference_batch_size = 32
    samples = inference(trajectory_length, forward_model, env, inference_batch_size,
                        min_policy_std, max_policy_std, trajectory_length)
    samples = samples.cpu().numpy()
    if debug:
        fom = torch.rand(samples.shape[0])
    else:
        fom = compute_FOM_parallele(samples)
    report_objective(fom.max(), "max FOM")
    np.save(os.path.join(savepath, 'samples.npy'), samples)
    np.save(os.path.join(savepath, 'fom.npy'), fom)
    plot_samples_and_histogram(samples, fom, 10, savepath)

    ic(torch.cuda.max_memory_allocated()/1024**2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store_true', default=False)
    parser.add_argument('-lr_model', type=float, default=1e-3)
    parser.add_argument('-lr_logz', type=float, default=1e-1)
    parser.add_argument('-min_policy_std', type=float, default=0.1)
    parser.add_argument('-max_policy_std', type=float, default=1.)
    args = parser.parse_args()
    debug = args.d
    # make args into a dict
    setup_configs = vars(args)

    main(debug, **setup_configs)
