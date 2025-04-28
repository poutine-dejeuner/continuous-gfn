import torch

from tuto import action_parameters_to_state_space

def plot(images):


def test__action_parameters_to_state_space():
    action = torch.rand(2, 6)
    action = action_parameters_to_state_space(action)

