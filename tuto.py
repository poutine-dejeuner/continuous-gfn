from matplotlib import pyplot as plt
from torch.distributions import Normal
import math
import numpy as np
import torch
import random
from tqdm import trange

from gaussians import add_gaussians_to_images
from gflownet.utils.photo.utils import RBF, rbf_parameters_to_designs
from nanophoto.unet_fompred import get_unet_fompred
from nanophoto.models import ScatteringNetwork


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trajectory_length = 5
min_policy_std = 0.1
max_policy_std = 1.0
batch_size = 256

class PhotoEnv():
    def __init__(
        self,
        state_shape=(101,91),
        action_dim=5,
        reward_debug=False,
        device_str="gpu",
        verify_actions=False,
        pf_model=ScatteringNetwork(config.scatter),
        pb_model=ScatteringNetwork(config.scatter),
    ):
        # Set verify_actions to False to disable action verification for faster step execution.
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.device_str = device_str
        self.device = torch.device(device_str)
        self.terminal_action = torch.full((self.action_dim,), -float("inf"), device=self.device)
        self.sink_state = torch.full(self.state_shape, -float("inf"), device=self.device)
        self.verify_actions = verify_actions
        self.reward_fun = get_unet_fompred

        self.pf_model = pf_model
        self.pb_model = pb_model

    def reward(self, x):
        fompred = self.reward_fun(x)
        return fompred

    def get_policy_dist(self, model, x):
        """
        A policy is a distribution we predict the parameters of using a neural
        network, which we then sample from.
        """
        pf_params = model(x)  # Shape = [batch_shape, env.action_dim]
        policy_mean = pf_params[:, 0: self.action_dim]
        policy_std = torch.sigmoid(pf_params[:, 1]) * (max_policy_std - min_policy_std) + min_policy_std
        policy_dist = torch.distributions.Normal(policy_mean, policy_std)

        return policy_dist

