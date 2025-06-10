import os
import numpy as np
import torch
from icecream import ic
import matplotlib.pyplot as plt

from tuto4 import setup_experiment, get_policy_dist, PhotoEnv
from tutoutils import test_print_traj
from distributions import RBF, grid

np.set_printoptions(precision=2)


def test__model_sampling():
    threshold = 0.1
    batch_size = 8
    traj_len = 4
    state_shape = (101, 91)
    action_dim = 7
    device = torch.device("cuda")
    savepath = "outfiles/trajdebug"

    env = PhotoEnv()
    forward_model, backward_model, logZ, optimizer = setup_experiment()
    forward_model = forward_model.eval()
    x = torch.ones((batch_size,) + state_shape, device=torch.device("cuda")) * 1/2
    traj = torch.zeros((batch_size, traj_len, action_dim), device=device)
    for i in range(traj_len):
        # y = forward_model(x.unsqueeze(1))
        # ic(y.min(), y.max(), y.mean())
        policy_dist = get_policy_dist(env, forward_model, x, min_policy_std=0.1,
                max_policy_std=1)
        action = policy_dist.sample()        
        amp = action[:, -1]
        ic(amp.min(), amp.max(), amp.mean())
        traj[:, i, :] = action

        action = action.unsqueeze(1)
        image = RBF(traj)(grid((101, 91), (0, 1), (0, 1)).cuda())
        plt.imshow(image[:,:,0].cpu().numpy())
        plt.title(action[0].cpu().numpy())
        plt.savefig(os.path.join(savepath, f"action{i}.png"))

    os.makedirs(savepath, exist_ok=True)
    test_print_traj(traj, savepath, "test")    
    # test if all dim 0 slices of action are equal
    for i in range(1, batch_size):
        abs_err = torch.abs(action[i] - action[0]).mean()
        assert abs_err > threshold
    print("OK")

test__model_sampling()
