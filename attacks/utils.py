import torch


def normalized_random_init(shape: torch.Size, norm: str):
    """
    Args:
    - shape: shape of expected tensor. (B,C,H,W)
    - norm: type of norm
    """
    allowed_norms = 'linf l2'.split()
    assert len(shape) == 4
    assert norm in allowed_norms

    if norm == 'linf':
        init = 2.0 * torch.rand(shape, dtype=torch.float32, device='cuda') - 1.0
    elif norm == 'l2':
        init = 2.0 * torch.rand(shape, dtype=torch.float32, device='cuda') - 1.0
        init_norm = torch.norm(init.view(init.size(0), -1), p=2.0, dim=1)  # (B)
        normalized_init = init / init_norm[:, None, None, None]

        dim = init.size(1) * init.size(2) * init.size(3)
        rand_norm = torch.pow(torch.rand(init.size(0), dtype=torch.float32, device='cuda'), 1.0 / dim)
        init = normalized_init * rand_norm[:, None, None, None, None]
    else:
        raise NotImplementedError

    return init


class Normalizer(torch.nn.Module):
    def __init__(self, input_size, mean, std, from_pixel_space=True):
        """
        Args
        - input_size (int)	: input_sizeution of input image.
        - mean (tuple)	: mean of normalized pixel value of channels.
        - std (tuple)	: standard deviation of normalized pixel value of channels.
        """
        super().__init__()
        assert input_size > 0
        assert len(mean) > 0
        assert len(mean) == len(std)
        num_channel = len(mean)

        mean_list = [torch.full((input_size, input_size), mean[i], device='cuda') for i in range(num_channel)]
        self.mean = torch.unsqueeze(torch.stack(mean_list), 0)

        std_list = [torch.full((input_size, input_size), std[i], device='cuda') for i in range(num_channel)]
        self.std = torch.unsqueeze(torch.stack(std_list), 0)

        self.from_pixel_space = from_pixel_space

    def forward(self, x):
        """
        Args
        - x (torch.tensor) : tensor scales in pixel space [0, 255].
        """
        x = x / 255.0 if self.from_pixel_space else x
        x = x.sub(self.mean)
        x = x.div(self.std)
        return x


class Denormalizer(torch.nn.Module):
    def __init__(self, input_size, mean, std, to_pixel_space=True):
        """
        Args
        - input_size (int)	: input_sizeution of input image.
        - mean (tuple)	: mean of normalized pixel value of channels.
        - std (tuple)	: standard deviation of normalized pixel value of channels.
        """
        super().__init__()
        assert input_size > 0
        assert len(mean) > 0
        assert len(mean) == len(std)
        num_channel = len(mean)

        mean_list = [torch.full((input_size, input_size), mean[i], device='cuda') for i in range(num_channel)]
        self.mean = torch.unsqueeze(torch.stack(mean_list), 0)

        std_list = [torch.full((input_size, input_size), std[i], device='cuda') for i in range(num_channel)]
        self.std = torch.unsqueeze(torch.stack(std_list), 0)

        self.to_pixel_space = to_pixel_space

    def forward(self, x):
        """
        Args
        - x (torch.tensor) : tensor scales in [0, 1.0].
        """
        x = x.mul(self.std)
        x = x.add(self.mean)
        x = x * 255.0 if self.to_pixel_space else x
        return x


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1)  # top-k index: size (B, k)
        pred = pred.t()  # size (k, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            acc.append(correct_k * 100.0 / batch_size)
        return acc
