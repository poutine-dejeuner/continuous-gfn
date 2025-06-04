import numpy as np
import torch
import matplotlib.pyplot as plt

from icecream import ic

from tuto import action_parameters_to_state_space
from tutoutils import rbf

np.set_printoptions(precision=2)

def plot(images, N=2):
    _, axs = plt.subplots(N, N, figsize=(15, 5))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i])
        ax.axis('off')
    plt.show()


def test__rbf():
    params = torch.rand(4, 2, 6)
    images = rbf(params, image_shape=(101, 91))
    images = images > 0.5
    plot(images)


def test__action_parameters_to_state_space():
    action = torch.rand(8, 1, 6)
    action = action_parameters_to_state_space(action)
    # ic(action.shape)
    plot(action)

def get_policy_dist(action_dim, model, x, min_policy_std, max_policy_std):
    """
    A policy is a distribution we predict the parameters of using a neural
    network, which we then sample from.
    """
    x = x.unsqueeze(1)
    x = x.contiguous()
    pf_params = model(x)  # Shape = [batch_shape, action_dim]
    policy_mean = pf_params[:, 0: action_dim]
    policy_std = pf_params[:, action_dim:]
    policy_std = torch.clamp(policy_std, min_policy_std, max_policy_std)
    policy_dist = torch.distributions.Normal(policy_mean, policy_std)
    return policy_dist, pf_params

def step(x, action):
    """Takes a forward step in the environment."""
    new_x = torch.zeros_like(x)
    action = action_parameters_to_state_space(action).to(x.device)
    new_x = x + action
    new_x = torch.clip(new_x, min=0, max=1)
    return new_x

def test__trained_forward_model_output():
    print("trained model")
    from tutoutils import CustomResNet152

    device = torch.device("cuda")
    path = "outfiles/6825570/forwardm.pth"
    forward_model = CustomResNet152(out_features=12).to(device)
    forward_model_weights = torch.load(path)
    forward_model.load_state_dict(forward_model_weights)
    test__forward_model(forward_model)

def test__untrained_model():
    print("untrained model")
    from tutoutils import CustomResNet152

    device = torch.device("cuda")
    forward_model = CustomResNet152(out_features=12).to(device)
    test__forward_model(forward_model)

def test__forward_model(forward_model):
    device = next(forward_model.parameters()).device
    batch_size = 16
    trajectory_len = 32
    min_policy_std = 0.1
    max_policy_std = 1.
    action_dim = 6
    trajectory = []
    params = []

    with torch.no_grad():
        x = torch.zeros(batch_size, 101, 91, device=device)

        for _ in range(trajectory_len):
            policy_dist, pf_params = get_policy_dist(action_dim, forward_model, x, min_policy_std, max_policy_std)
            params.append(pf_params)
            action = policy_dist.sample()
            trajectory.append(action)

            x = step(x, action)

    x = torch.nn.functional.sigmoid(x)
    diff = 0
    for i in range(batch_size):
        for j in range(i, batch_size):
            diff += torch.sum(torch.abs(x[i] - x[j]))
    print("avg diff")
    # ic(diff/trajectory_len)
    trajectory = torch.stack(trajectory, dim=1)
    trajectory = trajectory.cpu().numpy()
    ic(trajectory.shape)
    # print(trajectory)
    mean = trajectory.mean(axis=(0,1))
    std = trajectory.std(axis=(0,1))

    ic(mean.shape, std.shape)
    ic(mean, std)


# test__trained_forward_model_output()
# test__untrained_model()
test__rbf()
# test__action_parameters_to_state_space()
