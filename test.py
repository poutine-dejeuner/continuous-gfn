import torch
import matplotlib.pyplot as plt

from icecream import ic

from tuto import action_parameters_to_state_space
from tutoutils import rbf

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
    ic(action.shape)

    plot(action)


test__rbf()
test__action_parameters_to_state_space()
