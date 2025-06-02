import torch
from icecream import ic
from tuto2 import setup_experiment, get_policy_dist, PhotoEnv

def test__model_sampling():
    env = PhotoEnv()
    forward_model, backward_model, logZ, optimizer = setup_experiment()
    x = torch.rand(16, 101, 91, device=torch.device("cuda"))
    y = forward_model(x.unsqueeze(1))
    policy_dist = get_policy_dist(env, forward_model, x, min_policy_std=0.1,
            max_policy_std=1)
    action = policy_dist.sample()
    # test if all dim 0 slices of action are equal
    for i in range(1, action.shape[0]):
        assert not torch.all(action[i] == action[0])

test__model_sampling()
