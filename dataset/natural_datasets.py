import os

import cv2
import torch
from PIL import Image
from tqdm import trange

from dataset.basedataset import BaseDataSet
from dataset.splitter import *
from dataset.transforms import *

"""
Improvements:
1. Use fix scale in validation
2. Base_size is 1024(short edge is used)
3. Fill -1(ignore class) in mask
4. base_size is long size
5. align_corners = True
6. lr x10 for head

1. Network checked!
2. Dataset checked!
3. scheduler checked!
4. 
"""

ignore_idx = -1


def get_transform(mode='train', base_long_size=520, scale=[0.5, 2.0],
                  crop_size=480, do_random_scale=True, random_rotate=False, val_full_size=False):
    scale_op = RandomScale(base_long_size, scale) if do_random_scale else do_nothing()
    if mode == 'train':
        if random_rotate:
            transforms = Compose([
                RandomFlip(),
                scale_op,
                RandomRotate(angle_range=(-10, 10)),
                RandomCrop([crop_size, crop_size], mask_fill=ignore_idx),
                random_gaussian_blur(),
            ])
        else:
            transforms = Compose([
                RandomFlip(),
                scale_op,
                RandomCrop([crop_size, crop_size], mask_fill=ignore_idx),
                random_gaussian_blur(),
            ])

    elif mode == 'val' and not val_full_size:
        transforms = Compose([
            FixScale(short_size=crop_size),
            # MARK : test if do not center crop when validate or test
            CenterCrop([crop_size, crop_size], mask_fill=ignore_idx),
        ])
    else:
        transforms = None
    return transforms


from utils.visualize import show_graphs

labels = \
    [
        ('void', 255, (0, 0, 0)),
        ('road', 0, (128, 64, 128)),
        ('sidewalk', 1, (244, 35, 232)),
        ('building', 2, (70, 70, 70)),
        ('wall', 3, (102, 102, 156)),
        ('fence', 4, (190, 153, 153)),
        ('pole', 5, (153, 153, 153)),
        ('traffic_light', 6, (250, 170, 30)),
        ('traffic_sign', 7, (220, 220, 0)),
        ('vegetation', 8, (107, 142, 35)),
        ('terrain', 9, (152, 251, 152)),
        ('sky', 10, (70, 130, 180)),
        ('person', 11, (220, 20, 60)),
        ('rider', 12, (255, 0, 0)),
        ('car', 13, (0, 0, 142)),
        ('truck', 14, (0, 0, 70)),
        ('bus', 15, (0, 60, 100)),
        ('train', 16, (0, 80, 100)),
        ('motorcycle', 17, (0, 0, 230)),
        ('bicycle', 18, (119, 11, 32)),
    ]


def color_map_2_class_map(rgb_mask):
    # [G, R, B]
    H, W, C = rgb_mask.shape
    mask = np.zeros((H, W)) - 1
    for label in labels:
        rgb = label[2]
        id = label[1]
        idx = ((rgb_mask[:, :, 0] == rgb[2]) & (rgb_mask[:, :, 1] == rgb[1]) & (rgb_mask[:, :, 2] == rgb[0]))
        mask[idx] = id
    # can not be processed in PIL image
    mask[mask == -1] = 255
    return mask


def class_map_2_color_map(class_map):
    # [R, G, B]
    H, W = class_map.shape
    mask = np.zeros((3, H, W))
    for i in range(19):
        mask[:, class_map == i] = np.array(labels[i + 1][2]).reshape(3, 1)
    return mask


# translate other key_class_map to cityscapes key_class_map
def label_to_citys(classes, citys_idx_map):
    converted = 0
    final_map = []
    for clz in classes:
        if clz in citys_idx_map:
            # if has same key, translate this key's id to citys_id
            city_id = citys_idx_map[clz]
            converted += 1
        else:
            city_id = ignore_idx
        final_map.append(city_id)
    # print('converted {} classes'.format(converted))
    return final_map


# 2048 x 1080
# 2975, 500, 1525,
# 120 epoch
class CityScapesDataSet(BaseDataSet):
    nclass = 19

    def __init__(self, root, output_path='.', force_cache=False, mode='train',
                 base_size=2048, crop_size=768, scale=[0.5, 2.0], random_scale=True,
                 random_rotate=True, val_full_size=False):
        transforms = get_transform(mode=mode, base_long_size=base_size, crop_size=crop_size,
                                   scale=scale, do_random_scale=random_scale, random_rotate=random_rotate,
                                   val_full_size=val_full_size)
        super(CityScapesDataSet, self).__init__(root, output_path=output_path, force_cache=force_cache,
                                                transform=transforms, mode=mode)
        objects = {'road': 7, 'sidewalk': 8, 'building': 11, 'wall': 12, 'fence': 13, 'pole': 17, 'traffic_light': 19,
                   'traffic_sign': 20, 'vegetation': 21, 'terrain': 22, 'sky': 23, 'person': 24, 'rider': 25,
                   'car': 26, 'truck': 27, 'bus': 28, 'train': 31, 'motorcycle': 32, 'bicycle': 33}
        self.idx_of_objects = {k: i for i, k in enumerate(objects.keys())}
        self.idx_2_key = {i:k for i, k in enumerate(objects.keys())}
        self._key = np.array([ignore_idx,
                              ignore_idx, ignore_idx, ignore_idx, ignore_idx, ignore_idx,
                              ignore_idx, ignore_idx, 0, 1, ignore_idx, ignore_idx,
                              2, 3, 4, ignore_idx, ignore_idx, ignore_idx,
                              5, ignore_idx, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              ignore_idx, ignore_idx, 16, 17, 18])
        # mapping : (-1, 33) = 35
        self._mapping = np.array(range(ignore_idx, len(self._key) - 1)).astype('int32')

    def print_label_mapping(self):
        for c, id in self.idx_of_objects.items():
            print('{} : {}'.format(c, id), end=' ')
        print('')
        print('Class range : [0, 33], (no 255)')
        print(self._key)
        print(self._mapping)

    def split_data(self, root):
        root = Path(root) / 'leftImg8bit'
        img_to_mask = [['leftImg8bit', '.png'], ['gtFine', '_labelIds.png']]
        splitter = TwoLevelSplitter(root, img_to_mask, cache_path=self.output_path / 'dataset' / 'cityscapes',
                                    # TODO change back
                                    # test_name='val',
                                    img_filter='png', force_cache=self.force_cache)
        train, dev, test = splitter.get_train_dev_test_path()
        return train, dev, test

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for v in values:
            assert (v in self._mapping), '{} is not in mapping {}'.format(v, self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        result = self._key[index].reshape(mask.shape)
        return result

    def _index_to_class(self, trainId_prediction):
        values = np.unique(trainId_prediction)
        for v in values:
            assert (v in self._key), '{} is not in mapping {}'.format(v, self._mapping)

        value, key = np.array(self._mapping), np.array(self._key)
        arg = np.argsort(key)
        value, key = value[arg], key[arg]

        index = np.digitize(trainId_prediction.ravel(), key, right=True)
        labelId_prediction = value[index].reshape(trainId_prediction.shape)
        return labelId_prediction.astype(np.uint8)

    def post_process(self, img, mask, mode):
        # target = mask.numpy()
        if mask is None:
            return img, mask
        mask[mask == 255] = -1
        target = self._class_to_index(np.array(mask).astype('int32'))
        mask = torch.from_numpy(target).long()
        return img, mask

    def predict(self, trainId_prediction):
        return self._index_to_class(trainId_prediction)

    def format_class_iou(self, IoU_list):
        string = ''
        for i, iou in enumerate(IoU_list):
            string += '{} : {:.4f}  '.format(self.idx_2_key[i], iou)
        return string

# 1280 x 760
# 6580, 500, 500
# 60 epoch
class Synthia(CityScapesDataSet):
    def __init__(self, root, output_path='.', force_cache=False, mode='train',
                 base_size=1280, crop_size=760, scale=[0.5, 2.0], random_scale=True, random_rotate=True):
        super(Synthia, self).__init__(root, output_path, force_cache, mode, base_size, crop_size, scale, random_scale, random_rotate)
        self.classes = ['void', 'void', 'sky', 'building', 'road', 'sidewalk', 'fence', 'vegetation', 'pole', 'car',
                        'traffic_sign',
                        'person', 'bicycle', 'motorcycle', 'parking_slot', 'road_work', 'traffic_light', 'terrain',
                        'rider', 'truck', 'bus', 'train', 'wall', 'lanemarking']
        self._key = np.array(label_to_citys(self.classes, self.idx_of_objects)).astype('int32')
        self._mapping = np.array(range(-1, len(self._key))).astype('int32')

    def print_label_mapping(self):
        for c, id in zip(self.classes, self._key):
            print('{} : {}'.format(c, id), end=' ')
        print('')
        print('Class range : [0, 22]')
        print(self._key)
        print(self._mapping)

    def load_img(self, img_filename, mask_filename, mode):
        img = Image.open(img_filename)
        img = img.convert('RGB')

        mask = cv2.imread(mask_filename, -1)[:, :, -1]
        mask = Image.fromarray(mask)
        mask = mask.convert('L')
        return img, mask

    def split_data(self, root):
        root = Path(root) / 'RAND_CITYSCAPES' / 'RGB'
        img_to_mask = [['RGB'], ['GT/LABELS']]
        splitter = ZeroLevelKFoldSplitter(root, img_to_mask, cache_path=self.output_path / 'dataset' / 'synthia',
                                          train_rate=0.7, dev_rate=0.3,
                                          img_filter='png', force_cache=self.force_cache)
        train, dev, test = splitter.get_train_dev_test_path()
        return train, dev, dev

    def __len__(self):
        if self.mode == 'train':
            return len(self.paths)
        else:
            return 500


# 1914 x 1052
# 12403, 6382, 6181
# 30 epoch
class GTA5(CityScapesDataSet):
    def __init__(self, root, output_path='.', force_cache=False, mode='train',
                 base_size=1914, crop_size=700, scale=[0.5, 2.0], random_scale=True,
                 random_rotate=True, dataset_name='gta5'):
        self.name = dataset_name
        super(GTA5, self).__init__(root, output_path, force_cache, mode, base_size, crop_size, scale, random_scale, random_rotate)
        self.classes = [label[0] for label in labels]
        self._key = np.array(label_to_citys(self.classes, self.idx_of_objects)).astype('int32')
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')
        # self.print_label_mapping()

    def print_label_mapping(self):
        for c, id in zip(self.classes, self._key):
            print('{} : {}'.format(c, id), end=' ')
        print('')
        print('Class range : [-1, 18], (255==-1)')
        print(self._key)
        print(self._mapping)

    def load_img(self, img_filename, mask_filename, mode):
        img = Image.open(img_filename)
        img = img.convert('RGB')

        rgb_mask = cv2.imread(mask_filename, -1)
        if rgb_mask is None:
            mask = np.zeros_like(np.array(img))
        else:
            mask = color_map_2_class_map(rgb_mask)
        mask = Image.fromarray(mask)
        mask = mask.convert('L')
        return img, mask

    def split_data(self, root):
        root = Path(root)
        img_to_mask = [['images'], ['labels']]
        splitter = MatSplitter(root, root / 'split.mat', 'images', img_to_mask,
                               cache_path=self.output_path / 'dataset' / self.name,
                               img_filter='png', force_cache=self.force_cache)
        train, dev, test = splitter.get_train_dev_test_path()
        return train, dev, test

    def __len__(self):
        if self.mode == 'train':
            return len(self.paths)
        else:
            return 500


class GTA5_Multi(GTA5):
    def __init__(self, gta5_root, **kwargs):
        super(GTA5_Multi, self).__init__(**kwargs)
        self.gta5_root = Path(gta5_root)

    def split_data(self, root):
        root = Path(root)
        img_to_mask = [[str(root)], [str(self.gta5_root/'labels')]]
        splitter = MatSplitter(root, self.gta5_root / 'split.mat', '', img_to_mask,
                               cache_path=self.output_path / 'dataset' / self.name,
                               img_filter='png', force_cache=self.force_cache)
        train, dev, test = splitter.get_train_dev_test_path()
        return train, dev, test


# 2592 x 1936
# 17992, 2000, 5000
class Mapillary(GTA5):
    def __init__(self, root, output_path='.', force_cache=False, mode='train',
                 base_size=2590, crop_size=769, scale=[0.5, 2.0], random_scale=True, random_rotate=True):
        super(Mapillary, self).__init__(root, output_path, force_cache, mode, base_size, crop_size, scale, random_scale, random_rotate)

    def split_data(self, root):
        root = Path(root)
        img_to_mask = [['images', '.jpg'], ['labels', '.png']]
        splitter = ThreeLevelSplitter(root, img_to_mask, cache_path=self.output_path / 'dataset' / 'mapillary',
                                      middle_folder='images',
                                      train_name='training', val_name='validation', test_name='testing',
                                      img_filter='jpg', force_cache=self.force_cache)
        train, dev, test = splitter.get_train_dev_test_path()
        return train, dev, test


# 1280 x 964
# 6993, 981, 2029,
class IDD(CityScapesDataSet):
    def __init__(self, root, output_path='.', force_cache=False, mode='train',
                 base_size=1280, crop_size=768, scale=[0.5, 2.0], random_scale=True, random_rotate=True):
        super(IDD, self).__init__(root, output_path, force_cache, mode, base_size, crop_size, scale, random_scale, random_rotate)
        # self.print_label_mapping()

    def split_data(self, root):
        root = Path(root) / 'leftImg8bit'
        img_to_mask = [['leftImg8bit', '.png'], ['gtFine', '_labelcsIds.png']]
        splitter = TwoLevelSplitter(root, img_to_mask, cache_path=self.output_path / 'dataset' / 'IDD',
                                    img_filter='png', force_cache=self.force_cache)
        train, dev, test = splitter.get_train_dev_test_path()
        return train, dev, test

    def print_label_mapping(self):
        for c, id in self.idx_of_objects.items():
            print('{} : {}'.format(c, id), end=' ')
        print('')
        print('Class range : [0, 33], (has 255)')
        print(self._key)
        print(self._mapping)


def to_gray(img):
    return (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2])[None]


def label_consistency(label1, label2):
    for i in range(19):
        l1 = label1.clone()
        l2 = label2.clone()
        l1[l1 != i] = 255
        l2[l2 != i] = 255
        show_graphs([l1, l2])


def dataset_test(dataset):
    from utils.visualize import show_graphs
    for i, (name, img, mask) in enumerate(dataset):
        print(name)
        img = np.array(img)
        mask = class_map_2_color_map(np.array(mask[0]))
        show_graphs([img, mask], [name.split('/')[-1][-16:], 'mask'])
        if i == 1:
            break


def index_class_conversion():
    dataset = CityScapesDataSet(root=ROOT+'cityscapes', force_cache=True)
    claz = np.array(range(34))
    index = dataset._class_to_index(claz)
    new_claz = dataset._index_to_class(index)
    print(claz)
    print(index)
    print(new_claz)


if __name__ == '__main__':
    ROOT = '/data/DataSets/'
    roots = [ROOT + 'cityscapes',
             ROOT + 'SYNTHIA', ROOT + 'GTA5',
             ROOT + 'mapillary', ROOT + 'IDD', ]
    datasets = [CityScapesDataSet, Synthia, GTA5, Mapillary, IDD,
                ]
    output_path = 'tmp'
    force_cache = True
    idx = 5
    modes = ['train', 'val', 'test']

    for root, dataset in zip(roots[idx:idx + 5], datasets[idx:idx + 5]):
        print(root)
        d = dataset(root, output_path, force_cache, 'train')
        dataset_test(d)
