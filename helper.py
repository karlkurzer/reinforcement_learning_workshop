from enum import Enum
from typing import List, Tuple

import gymnasium
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ALPHA_VALUE = 0.7
TEXT_FONT_SIZE = 10


def create_environment(env_name: str):
    """Create a Gymnasium environment.

    Args:
        env_name (str): Name of the environment

    Returns:
        gymnasium.Env: Gymnasium environment
    """
    env = gymnasium.make(
        env_name,
        is_slippery=False,
        render_mode="rgb_array",
    )
    return env


def generate_checkerboard(
    img: np.ndarray, v: np.ndarray
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Generates a checkerboard pattern by mapping matrix V onto image img.
    """
    size_y, size_x = img.shape[:2]
    interpolation_factor_y = size_y // v.shape[0]
    interpolation_factor_x = size_x // v.shape[1]

    # Broadcasting the smaller matrix V to the size of img
    checkerboard = np.repeat(
        np.repeat(v, interpolation_factor_y, axis=0), interpolation_factor_x, axis=1
    )

    return checkerboard, (interpolation_factor_x, interpolation_factor_y)


def generate_colormap():
    """
    Generates a default colormap.
    """
    return LinearSegmentedColormap.from_list(
        "custom_cmap", [(0, "red"), (0.5, "white"), (1, "green")]
    )


def add_labels(
    ax, shape: Tuple[int, int], labels: List, interpolation_factors: Tuple[int, int]
):
    """
    Adds labels to the cells of the visualization.
    Parameters:
        ax (matplotlib.axes.Axes): The axes on which to add labels.
        shape (tuple): The shape of the grid.
        labels (list): The labels to add.
        interpolation_factors (tuple): The interpolation factors for positioning labels.
    """
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax.text(
                interpolation_factors[0] * (j + 0.5),
                interpolation_factors[1] * (i + 0.5),
                labels[i, j],
                ha="center",
                va="center",
                fontsize=TEXT_FONT_SIZE,
                fontweight="bold",
                alpha=ALPHA_VALUE,
            )


def visualize_v(env, v: np.ndarray, ax, title: str) -> None:
    """Visualizes the value function v of the given environment."""
    v = v.reshape(env.unwrapped.desc.shape)
    v_img, interp_factors = generate_checkerboard(env.render(), v)
    v_img = ax.imshow(v_img, cmap=generate_colormap(), alpha=0.5)

    labels = np.vectorize(lambda x: f"{x:.2f}")(v)

    add_labels(ax, env.unwrapped.desc.shape, labels, interp_factors)

    visualize_env(env, ax, title)


def visualize_p(
    env, v: np.ndarray, p: np.ndarray, action: Enum, ax, title: str
) -> None:
    """Visualizes the policy p and the of the given environment."""
    v = v.reshape(env.unwrapped.desc.shape)
    v_img, interp_factors = generate_checkerboard(env.render(), v)
    v_img = ax.imshow(v_img, cmap=generate_colormap(), alpha=0.5)

    s = np.arange(env.unwrapped.observation_space.n)
    labels = np.vectorize(lambda x: action(p[x]).name)(s).reshape(
        env.unwrapped.desc.shape
    )

    add_labels(ax, env.unwrapped.desc.shape, labels, interp_factors)

    visualize_env(env, ax, title)


def visualize_env(env, ax, title: str) -> None:
    """
    Visualizes the FrozenLake environment.
    """
    ax.imshow(env.render(), alpha=ALPHA_VALUE)

    ax.axis("off")
    ax.set_title(title)


class DPType(Enum):
    """
    Enum for the different types of dynamic programming.
    """

    VALUE_ITERATION = 0
    POLICY_ITERATION = 1
