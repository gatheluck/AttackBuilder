import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import random
import torch

from attacks.attack import AttackWrapper
from attacks.utils import normalized_random_init


class PgdAttack(AttackWrapper):
    SUPPORTED_NORM = 'linf l2'.split()

    def __init__(self, input_size: int, mean: tuple, std: tuple, num_iteration: int, eps_max: float, step_size: float, norm: str, rand_init: bool, scale_each: bool, criterion=torch.nn.CrossEntropyLoss(), device='cuda'):
        """
        Args
        - input_size (int)
        - mean (tuple)
        - std (tuple)
        - num_iteration (int)
        - eps_max (float): (max) size of perturbation
        - step_size (float): step size of single iteration
        - norm (str): type of norm
        - rand_init (bool): whether execute random init of delta or not
        - scale_each (bool): whether to scale eps for each image in a batch separately. useful for adversarial training
        """
        super().__init__(input_size=input_size, mean=mean, std=std)
        if norm not in self.SUPPORTED_NORM:
            raise ValueError('not supported norm type')

        assert num_iteration >= 0
        assert eps_max >= 0.0
        assert step_size >= 0.0

        self.num_iteration = num_iteration
        self.eps_max = eps_max
        self.step_size = step_size
        self.norm = norm
        self.rand_init = rand_init
        self.scale_each = scale_each
        self.device = device
        self.criterion = criterion.cuda() if self.device == 'cuda' else criterion

    def _init_delta(self, shape, eps):
        """
        initialize delta. if self.rand_init is True, execute random initialization.
        """
        if self.rand_init:
            init_delta = normalized_random_init(shape, self.norm, device=self.device)  # initialize delta for linf or l2

            init_delta = eps[:, None, None, None] * init_delta  # scale by eps
            init_delta.requires_grad_()
            return init_delta
        else:
            return torch.zeros(shape, requires_grad=True, device=self.device)

    def _forward(self, pixel_model, pixel_x: torch.tensor, target, avoid_target: bool, scale_eps: bool):
        # if scale_eps is True, change eps adaptively. this usually improve robustness against wide range of attack
        if scale_eps:
            if self.scale_each:
                rand = torch.rand(pixel_x.size()[0], device=self.device)
            else:
                rand = random.random() * torch.ones(pixel_x.size()[0], device=self.device)
            base_eps = rand.mul(self.eps_max)
            step_size = rand.mul(self.step_size)
        else:
            base_eps = self.eps_max * torch.ones(pixel_x.size(0), device=self.device)
            step_size = self.step_size * torch.ones(pixel_x.size(0), device=self.device)

        # init delta
        pixel_input = pixel_x.detach()
        pixel_input.requires_grad_()
        pixel_delta = self._init_delta(pixel_input.size(), base_eps)

        # compute delta in pixel space
        if self.num_iteration:  # run iteration
            pixel_delta = self._run(pixel_model, pixel_input, pixel_delta, target, avoid_target, base_eps, step_size)
        else:  # if self.num_iteration is 0, return just initialization result
            pixel_delta.data = torch.clamp(pixel_input.data, + pixel_delta.data, 0.0, 255.0) - pixel_input.data

        # IMPORTANT: this return is in PIXEL SPACE (=[0,255])
        return pixel_input + pixel_delta

    def _run(self, pixel_model, pixel_input, pixel_delta, target, avoid_target, eps, step_size: torch.tensor):
        assert self.num_iteration > 0

        logit = pixel_model(pixel_input + pixel_delta)

        for it in range(self.num_iteration):
            loss = self.criterion(logit, target)
            loss.backward()

            if avoid_target:
                grad = pixel_delta.grad.data  # to avoid target, increase the loss
            else:
                grad = -pixel_delta.grad.data  # to hit target, decrease the loss

            if self.norm == 'linf':
                grad_sign = grad.sign()
                pixel_delta.data = pixel_delta.data + step_size[:, None, None, None] * grad_sign
                pixel_delta.data = torch.max(torch.min(pixel_delta.data, eps[:, None, None, None]), -eps[:, None, None, None])  # scale in [-eps, +eps]
                pixel_delta.data = torch.clamp(pixel_input.data + pixel_delta.data, 0.0, 255.0) - pixel_input.data
            else:
                raise NotImplementedError

            if it != self.num_iteration - 1:
                logit = pixel_model(pixel_input + pixel_delta)
                pixel_delta.grad.data.zero_()

        return pixel_delta


if __name__ == '__main__':
    import math
    import tqdm
    import collections
    import torchvision

    from attacks.utils import accuracy
    from attacks.utils import Denormalizer

    from submodules.DatasetBuilder.dataset_builder import DatasetBuilder
    from submodules.ModelBuilder.model_builder import ModelBuilder

    os.makedirs('../logs', exist_ok=True)
    # path = '/home/gatheluck/Scratch/Stronghold/logs/train/2020-06-26_12-55-57_cifar10/checkpoint/model_weight_final.pth'
    path = '/media/gatheluck/gathe-drive/models/uar/adv-cifar10-models/models/clean_model.pth'

    model = ModelBuilder(num_classes=10)['resnet56']
    model.load_state_dict(torch.load(path))
    model = model.cuda()
    model.eval()

    eps = 16.0
    num_iteration = 20
    # step_size = eps / math.sqrt(num_iteration)
    step_size = eps
    rand_init = False
    scale_eps = False
    scale_each = False
    mean = (0.49139968, 0.48215841, 0.44653091)
    std = (0.24703223, 0.24348513, 0.26158784)
    dataset_builder = DatasetBuilder(name='cifar10', input_size=32, mean=mean, std=std, root_path='../data')
    val_dataset = dataset_builder(train=False, normalize=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=256, shuffle=False)

    denormalizer = Denormalizer(32, mean, std, to_pixel_space=False)

    stdacc1_list = list()
    advacc1_list = list()

    with tqdm.tqdm(enumerate(val_loader)) as pbar:
        for i, (x, y) in pbar:
            x, y = x.to('cuda'), y.to('cuda')

            attacker = PgdAttack(32, mean, std, num_iteration, eps, step_size, 'linf', rand_init, scale_each)
            x_adv = attacker(model, x, target=y, avoid_target=True, scale_eps=scale_eps)
            y_predict_std = model(x)
            y_predict_adv = model(x_adv)

            stdacc1_list.append(*accuracy(y_predict_std, y))
            advacc1_list.append(*accuracy(y_predict_adv, y))
            pbar.set_postfix(collections.OrderedDict(std='{}'.format(*accuracy(y_predict_std, y)), adv='{}'.format(*accuracy(y_predict_adv, y))))

            if i == 0:

                x_for_save = torch.cat([denormalizer(x[0:8, :, :, :]), denormalizer(x_adv[0:8, :, :, :])], dim=2)
                torchvision.utils.save_image(x_for_save, '../logs/pgd_test.png')

    stdacc1 = sum(stdacc1_list) / float(len(stdacc1_list))
    advacc1 = sum(advacc1_list) / float(len(advacc1_list))

    print('std: {std}, adv: {adv}'.format(std=stdacc1, adv=advacc1))
