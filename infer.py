import sys
import os
import yaml

import numpy as np
import torch
from icecream import ic

from nanophoto.models import ScatteringNetwork
from nanophoto.meep_compute_fom import compute_FOM_parallele
from tuto import inference, plot_samples_and_histogram, PhotoEnv


def gen_samples_print_plots(env, forward_model, trajectory_length, savepath):
    min_policy_std = 0.1
    max_policy_std = 1.0
    batch_size = 32
    samples = inference(trajectory_length, forward_model, env, batch_size,
            min_policy_std, max_policy_std)
    samples = samples.cpu().numpy()
    ic(samples.min(), samples.max())

    fom = compute_FOM_parallele(samples)
    np.save(os.path.join(savepath, 'samples.npy'), samples)
    np.save(os.path.join(savepath, 'fom.npy'), fom)
    plot_samples_and_histogram(samples, fom, 10)


if __name__ == "__main__":
    directory = sys.argv[1] 
    ic(directory)
    trajectory_length = 20
    # from directory read the config.yml file and load the parameters
    with open(os.path.join(directory, 'config.yml'), 'r') as f:
        config = yaml.safe_load(f)
    env = PhotoEnv()
    model = ScatteringNetwork(**config, device=torch.device('cuda'),
            dtype=torch.float)
    state_dict = torch.load(os.path.join(directory, "forwardm.pth"))
    model.load_state_dict(state_dict)
    model.eval()
    gen_samples_print_plots(env, model, trajectory_length, directory)
