import os
import sys

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)

import hydra
import omegaconf

from attacks.pgd import PgdAttack


class AttackBuilder(object):
    SUPPORTED_ATTACK = 'pgd'.split()

    def __init__(self, name):
        if name not in self.SUPPORTED_ATTACK:
            raise ValueError
        self.name = name

    def __call__(self, **kwargs):

        if self.name == 'pgd':
            attacker = PgdAttack(**kwargs)
        else:
            raise NotImplementedError

        return attacker


@hydra.main(config_path='./conf/test.yaml')
def main(cfg: omegaconf.DictConfig) -> None:
    import os
    import math
    import tqdm
    import logging
    import collections
    import torch
    import torchvision

    from attacks.utils import accuracy
    from attacks.utils import Denormalizer

    from submodules.DatasetBuilder.dataset_builder import DatasetBuilder
    from submodules.ModelBuilder.model_builder import ModelBuilder

    logger = logging.getLogger(__name__)
    logger.info(cfg.pretty())

    cfg.attack.step_size = eval(cfg.attack.step_size)  # eval actual value of step_size

    model = ModelBuilder(num_classes=cfg.dataset.num_classes)[cfg.arch]
    model.load_state_dict(torch.load(os.path.join(hydra.utils.get_original_cwd(), cfg.weight)))
    model = model.cuda() if cfg.device == 'cuda' else model
    model.eval()

    dataset_builder = DatasetBuilder(root_path=os.path.join(hydra.utils.get_original_cwd(), './data'), **cfg.dataset)
    val_dataset = dataset_builder(train=False, normalize=cfg.normalize)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False)

    denormalizer = Denormalizer(cfg.dataset.input_size, cfg.dataset.mean, cfg.dataset.std, to_pixel_space=False)

    stdacc1_list = list()
    advacc1_list = list()

    with tqdm.tqdm(enumerate(val_loader)) as pbar:
        for i, (x, y) in pbar:
            x, y = x.to(cfg.device), y.to(cfg.device)

            # attacker = PgdAttack(cfg.dataset.input_size, cfg.dataset.mean, cfg.dataset.std, cfg.attack.num_iteration, cfg.attack.eps, cfg.attack.step_size, cfg.attack.norm, cfg.attack.rand_init, cfg.attack.scale_each)
            attacker = AttackBuilder('pgd')(input_size=cfg.dataset.input_size, mean=cfg.dataset.mean, std=cfg.dataset.std, **cfg.attack)
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

    logger.info('std: {std}, adv: {adv}'.format(std=stdacc1, adv=advacc1))


if __name__ == '__main__':
    main()