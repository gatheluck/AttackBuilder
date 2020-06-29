import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import hydra
import omegaconf
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
        assert eps_max > 0.0
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
        if self.norm == 'l2':
            l2_eps_max = eps

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
            elif self.norm == 'l2':
                batch_size = pixel_delta.data.size(0)
                grad_norm = torch.norm(grad.view(batch_size, -1), p=2.0, dim=1)  # IMPORTANT: if you set eps = 0.0 this leads nan
                normalized_grad = grad / grad_norm[:, None, None, None]
                pixel_delta.data = pixel_delta.data + step_size[:, None, None, None] * normalized_grad
                l2_pixel_delta = torch.norm(pixel_delta.data.view(batch_size, -1), p=2.0, dim=1)
                # check numerical instabitily
                proj_scale = torch.min(torch.ones_like(l2_pixel_delta, device=self.device), l2_eps_max / l2_pixel_delta)
                pixel_delta.data = pixel_delta.data * proj_scale[:, None, None, None]
                pixel_delta.data = torch.clamp(pixel_input.data + pixel_delta.data, 0.0, 255.0) - pixel_input.data
            else:
                raise NotImplementedError

            if it != self.num_iteration - 1:
                logit = pixel_model(pixel_input + pixel_delta)
                pixel_delta.grad.data.zero_()

        return pixel_delta


@hydra.main(config_path='../conf/test.yaml')
def main(cfg: omegaconf.DictConfig) -> None:
    import math
    import tqdm
    import logging
    import collections
    import torchvision

    from attacks.utils import accuracy
    from attacks.utils import Denormalizer

    from submodules.DatasetBuilder.dataset_builder import DatasetBuilder
    from submodules.ModelBuilder.model_builder import ModelBuilder

    logger = logging.getLogger(__name__)
    logger.info(cfg.pretty())

    # print(cfg.pretty())
    cfg.attack.step_size = eval(cfg.attack.step_size)  # eval actual value of step_size

    model = ModelBuilder(num_classes=cfg.dataset.num_classes)[cfg.arch]
    model.load_state_dict(torch.load(os.path.join(hydra.utils.get_original_cwd(), cfg.weight)))
    model = model.cuda() if cfg.device == 'cuda' else model
    model.eval()

    dataset_builder = DatasetBuilder(root_path=os.path.join(hydra.utils.get_original_cwd(), '../data'), **cfg.dataset)
    val_dataset = dataset_builder(train=False, normalize=cfg.normalize)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False)

    denormalizer = Denormalizer(cfg.dataset.input_size, cfg.dataset.mean, cfg.dataset.std, to_pixel_space=False)

    stdacc1_list = list()
    advacc1_list = list()

    with tqdm.tqdm(enumerate(val_loader)) as pbar:
        for i, (x, y) in pbar:
            x, y = x.to(cfg.device), y.to(cfg.device)

            attacker = PgdAttack(cfg.dataset.input_size, cfg.dataset.mean, cfg.dataset.std, cfg.attack.num_iteration, cfg.attack.eps, cfg.attack.step_size, cfg.attack.norm, cfg.attack.rand_init, cfg.attack.scale_each)
            x_adv = attacker(model, x, target=y, avoid_target=True, scale_eps=cfg.attack.scale_eps)
            with torch.autograd.no_grad():
                y_predict_std = model(x)
                y_predict_adv = model(x_adv)

            stdacc1_list.append(*accuracy(y_predict_std, y))
            advacc1_list.append(*accuracy(y_predict_adv, y))
            pbar.set_postfix(collections.OrderedDict(std='{}'.format(*accuracy(y_predict_std, y)), adv='{}'.format(*accuracy(y_predict_adv, y))))

            if i == 0:

                x_for_save = torch.cat([denormalizer(x[0:8, :, :, :]), denormalizer(x_adv[0:8, :, :, :])], dim=2)
                torchvision.utils.save_image(x_for_save, 'pgd_test.png')

    stdacc1 = sum(stdacc1_list) / float(len(stdacc1_list))
    advacc1 = sum(advacc1_list) / float(len(advacc1_list))

    # print('std: {std}, adv: {adv}'.format(std=stdacc1, adv=advacc1))
    logger.info('std: {std}, adv: {adv}'.format(std=stdacc1, adv=advacc1))


if __name__ == '__main__':
    main()
