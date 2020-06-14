import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch

from .attack import AttackWrapper
from .util import normalized_random_init


class PgdAttack(AttackWrapper):
    def __init__(self, input_size: int, mean: tuple, std: tuple, num_iteration: int, eps_max: float, step_size: float, norm: str, rand_init: bool):
        """
        Args
        - input_size (int)
        - mean (tuple)
        - std (tuple)
        - num_iteration (int)
        - eps_max (float)
        - rand_init (bool): whether execute random init of delta or not.
        """
        super().__init__(input_size=input_size, mean=mean, std=std)

    def _init_delta(self, shape, eps):
        """
        initialize delta. if self.rand_init is True, execute random initialization.
        """
        if self.rand_init:
            init_delta = normalized_random_init(shape, self.norm)
            init_delta = eps[:, None, None, None] * init_delta  # scale by eps
            init_delta.requires_grad_()
        else:
            return torch.zeros(shape, requires_grad=True, device='cuda')

    def _forward(self, pixel_model, pixel_x: torch.tensor, target: int, avoid_target: bool, scale_eps: bool):
        base_eps = self.eps_max * torch.ones(pixel_model.size(0), device='cuda')
        step_size = self.step_size * torch.ones(pixel_x.size(0), device='cuda')

        # init delta
        pixel_input = pixel_x.detach()
        pixel_input.requires_grad_()
        pixel_delta = self._init(pixel_input.size(), base_eps)

        # compute delta in pixel space
        if self.num_iteration:  # run iteration
            pixel_delta = self._run(pixel_model, pixel_input, pixel_delta, target, avoid_target, base_eps, step_size)
        else:  # return just initialization result
            pixel_delta.data = torch.clamp(pixel_input.data, + pixel_delta.data, 0.0, 255.0) - pixel_input.data

        return pixel_input + pixel_delta

    def _run(self, pixel_model, pixel_input, pixel_delta, target, avoid_target, eps, step_size):
        
        
