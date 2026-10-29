# SPDX-License-Identifier: MIT
# See LICENSE and NOTICE for copyright and source details.

"""Set seeds and devices, and update target networks."""

import copy
import random

import numpy as np
import torch


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    return seed


def set_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, int) or str(device).lstrip("-").isdigit():
        index = int(device)
        return torch.device(f"cuda:{index}" if index >= 0 and torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_target(model):
    target = copy.deepcopy(model)
    target.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def sync_target(source, target, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.copy_(target_param * (1 - tau) + source_param * tau)


def convert_to_tensor(value, device):
    return torch.as_tensor(value, device=device)


def noise_schedule(num_noises):
    """Set the noise levels used by the transition model."""
    t = np.arange(1, num_noises + 1)
    alphas = np.exp(-0.1 / num_noises - 0.5 * 9.9 * (2 * t - 1) / num_noises**2)
    return torch.as_tensor(np.cumprod(alphas), dtype=torch.float32)
