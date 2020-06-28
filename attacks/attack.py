import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from attacks.utils import Normalizer, Denormalizer

import torch


class PixelModel(torch.nn.Module):
    """

    """
    def __init__(self, model, input_size, mean, std):
        super().__init__()
        self.model = model
        self.normalizer = Normalizer(input_size, mean, std)

    def forward(self, pixel_x):
        """
        [0, 255] -> [0, 1]
        """
        x = self.normalizer(pixel_x)  # rescale [0, 255] -> [0, 1] and normalize
        # IMPORTANT: this return is in [0, 1]
        return self.model(x)


class AttackWrapper(torch.nn.Module):
    def __init__(self, **kwargs):
        """
        Args
        - input_size
        - mean
        - std
        """
        super().__init__()
        required_keys = set('input_size mean std'.split())
        parsed_args = self._parse_args(required_keys, kwargs)

        for k, v in parsed_args.items():
            setattr(self, k, v)

        self.normalizer = Normalizer(self.input_size, self.mean, self.std)
        self.denormalizer = Denormalizer(self.input_size, self.mean, self.std)

    def forward(self, model, x, *args, **kwargs):
        was_training = model.training
        pixel_model = PixelModel(model, self.input_size, self.mean, self.std)
        pixel_model.eval()
        # forward input to  pixel space
        pixel_x = self.denormalizer(x.detach())
        pixel_return = self._forward(pixel_model, pixel_x, *args, **kwargs)
        if was_training:
            pixel_model.train()

        return self.normalizer(pixel_return)

    @classmethod
    def _parse_args(cls, required_keys: set, input_args: dict) -> dict:
        """
        parse input args
        Args
        - required_keys (set) : set of required keys for input_args
        - input_args (dict)   : dict of input arugments
        """
        parsed_args = dict()

        for k in required_keys:
            if k not in input_args.keys():
                raise ValueError('initial args are invalid.')
            else:
                parsed_args[k] = input_args[k]

        return parsed_args