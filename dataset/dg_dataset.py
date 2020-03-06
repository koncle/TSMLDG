import functools

import numpy as np
import traceback
import torch
from torch.utils.data import DataLoader

from dataset.natural_datasets import *
from utils.visualize import show_graphs
from paths import *


def find_prefix(prefix, strings):
    find = False
    for string in strings:
        if prefix in string:
            find = True
            break
    return find, string


def get_dataset(name):
    dataset_dicts = {
        'G': GTA5,
        'S': Synthia,
        'I': IDD,
        'M': Mapillary,
        'C': CityScapesDataSet,
    }
    root_dicts = {
        'G': 'GTA5',
        'S': 'SYNTHIA',
        'I': 'IDD',
        'M': 'mapillary',
        'C': 'cityscapes',
    }
    if name not in dataset_dicts.keys():
        return None, None
    else:
        return dataset_dicts[name], root_dicts[name]


def get_target_loader(name, batch_size, mode='val', **kwargs):
    dataloader = functools.partial(DataLoader, num_workers=batch_size, pin_memory=True, batch_size=batch_size, **kwargs)
    target_dataset, folder = get_dataset(name)
    target_loader = dataloader(target_dataset(root=ROOT + folder, mode=mode))
    return target_loader


class DGMetaDataSets(object):
    def __init__(self, output_path='.', force_cache=True, mode='train',
                 crop_size=600, scale=[0.5, 2.0], random_scale=True,
                 random_rotate=True, post_processor=None, domains=['G', 'S'],
                 split_num=2):
        super(DGMetaDataSets, self).__init__()

        root = ROOT
        self.mode = mode
        self.domain_split_num = split_num

        # Generate domain datasets
        gta5_styles = {'c': 'cezanne', 'v': 'vangogh', 'u': 'ukiyoe'}
        included_styles = []
        self.domains = []
        for name in domains:
            dataset, folder = get_dataset(name)
            if dataset is None:
                if name in gta5_styles.keys():
                    included_styles.append(gta5_styles[name])
            self.domains.append(dataset(root + folder, output_path, force_cache, mode, crop_size=crop_size, scale=scale,
                                        random_scale=random_scale, random_rotate=random_rotate))  # , base_size=1024))

        aux_root = Path(Aux_ROOT)
        if aux_root.exists():
            for name in included_styles:
                folder = aux_root / ('style_' + name)
                gta5 = GTA5_Multi(root=str(folder), gta5_root=ROOT+'GTA5', output_path=output_path, force_cache=force_cache,
                                      mode=mode, crop_size=crop_size, scale=scale, random_scale=random_scale,
                                      dataset_name='gta5_' + name, random_rotate=random_rotate)
                self.domains.append(gta5)
        else:
            print(str(aux_root) + ' not exists, if you want to train with synthetic images, please generate it follow the README')

        print('domains {}, split_num {}, rotate {}, processor {}'.format(self.domains, split_num, random_rotate, post_processor))
        # post_processors
        self.post_processors = Compose([post_processor])

    def get_random_img_from(self, domain, idx, split_idx=None):
        while True:
            try:
                if self.mode == 'train':
                    if split_idx is None:
                        img_idx = np.random.randint(len(domain))  # train randomly
                    else:
                        segment_length = len(domain) // self.domain_split_num
                        img_idx = np.random.randint(segment_length * split_idx, segment_length * (split_idx + 1))
                else:
                    img_idx = idx  # test images in original order
                p, img, label = domain[img_idx]
                break
            except Exception as e:
                traceback.print_exc()

        return p, img, label

    def __getitem__(self, idx):
        paths, imgs, labels = [], [], []
        if len(self.domains) == 1:
            for i in range(self.domain_split_num):
                p, img, label = self.get_random_img_from(self.domains[0], idx, i)
                paths.append(p), imgs.append(img), labels.append(label)
        else:
            for i, (domain) in enumerate(self.domains):
                p, img, label = self.get_random_img_from(domain, idx, None)
                paths.append(p), imgs.append(img), labels.append(label)
        paths, imgs, labels = paths, torch.stack(imgs, 0), torch.stack(labels, 0)
        imgs, labels = self.post_processors(imgs, labels, self.mode)
        return paths, imgs, labels

    def __len__(self):
        if self.mode == 'train':
            return 3000
        else:
            return 500 // len(self.domains)


def show_dataset():
    # d1 = DGMetaDataSets(mode='val', domains=['gta5', 'synthia', 'mapillary', 'citys', 'idd'], )
    names = ['idd', 'gta5', 'synthia', 'mapillary', 'citys']
    for name in names:
        d1 = get_target_loader(name, 1, 'val', shuffle=True)
        print(name, len(d1))
        for i, (path, img, label) in enumerate(d1):
            for im, d_label in zip(img, label):
                show_graphs([im, class_map_2_color_map(d_label[0])])
                break
            break
