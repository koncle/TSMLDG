from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# __all__ = ['ZeroLevelKFoldSplitter', 'OneLevelKFoldSplitter',
#            'OneLevelSplitter', 'TwoLevelSplitter', 'ThreeLevelSplitter',
#             'CSVSplitter']


def split_trian_dev_test_path(paths, train_rate, dev_rate, test_rate=None):
    if test_rate is None:
        test_rate = 1 - train_rate - dev_rate
        assert test_rate >= -1e-5
    train_imgs, rest_imgs = train_test_split(paths, train_size=train_rate, shuffle=False)

    # train_rate = 0.8, dev_rate = 0.2, test_rate will be -1e-17, not 0
    if test_rate >= -1e-5 and test_rate <= 1e-5:
        dev_imgs = rest_imgs
        test_imgs = []
    else:
        dev_imgs, test_imgs = train_test_split(rest_imgs, train_size=dev_rate / (1 - train_rate), shuffle=False)
    return train_imgs, dev_imgs, test_imgs


class BaseSplitter(object):
    """Split the images file to train_net, dev, test file with only file path
    """

    def __init__(self, data_path, img_to_mask=None,
                 train_rate=0.6, dev_rate=0.2,
                 img_filter=None, not_img_filter=None, cache_path=None,
                 train_name='train', val_name='val', test_name='test',
                 force_cache=False):
        test_rate = 1 - train_rate - dev_rate
        assert test_rate >= -1e-5 or test_rate <= 1e-5
        self.img_to_mask = img_to_mask

        self.train_rate = train_rate
        self.dev_rate = dev_rate
        self.test_rate = test_rate

        self.cache_path = cache_path
        self.data_path = data_path
        if self.cache_path is None:
            self.cache_path = self.data_path
        self.force_cache = force_cache

        if isinstance(img_filter, (tuple, list)):
            self.filter = img_filter
        else:
            self.filter = [img_filter] if img_filter is not None else None

        if isinstance(not_img_filter, (tuple, list)):
            self.not_image_filter = not_img_filter
        else:
            self.not_image_filter = [not_img_filter] if not_img_filter is not None else None

        self.train_name = train_name
        self.val_name = val_name
        self.test_name = test_name

        self.train_data_path = []
        self.dev_data_path = []
        self.test_data_path = []
        self._split_file()

    def _split_file(self):
        if self.force_cache:
            self.split_file()
            self.sort_data()
            self.cache_file()
        else:
            flag = self._load_cache_file()
            if not flag:
                self.split_file()
                self.sort_data()
                self.cache_file()

    def split_file(self):
        raise NotImplementedError()

    def sort_data(self):
        self.train_data_path.sort()
        self.test_data_path.sort()
        self.dev_data_path.sort()

    def cache_file(self):
        path = Path(self.cache_path)
        from utils.nn_utils import mkdir
        mkdir(path, level=3)
        if path.is_dir():
            train = path / 'train.txt'
            dev = path / 'val.txt'
            test = path / 'test.txt'
            with train.open(mode='w', encoding='utf-8') as f:
                for path in self.train_data_path:
                    f.write(path + '\n')
            with dev.open(mode='w', encoding='utf-8') as f:
                for path in self.dev_data_path:
                    f.write(path + '\n')
            with test.open(mode='w', encoding='utf-8') as f:
                for path in self.test_data_path:
                    f.write(path + '\n')
        else:
            raise Exception("Wrong cache path")

    def add_file_to_data(self, file, path):
        if not (isinstance(file, list) or isinstance(file, np.ndarray)):
            files = [file]
        else:
            files = file

        for f in files:
            f = str(f)
            if self.filter is not None:
                # all required filter should be in the path
                if (np.array([str(f).find(filter) for filter in self.filter]) == -1).any():
                    continue

            if self.not_image_filter is not None:
                # no un required filter should be in the path
                if not (np.array([str(f).find(filter) for filter in self.not_image_filter]) == -1).all():
                    continue
            path.append(f + '\t' + self._get_mask_filename(f))

    def add_folder_to_data(self, folder, path):
        """ Filter no images with some pattern """
        folder = Path(folder)
        if not folder.exists():
            return
        paths = list(folder.iterdir())
        paths.sort()
        for img_path in paths:
            img_path = str(img_path)
            self.add_file_to_data(img_path, path)

    def get_train_dev_test_path(self):
        return self.train_data_path, self.dev_data_path, self.test_data_path

    def _get_mask_filename(self, img_filename):
        assert isinstance(self.img_to_mask[0], list) or isinstance(self.img_to_mask[0], tuple)
        img_strs, mask_strs = self.img_to_mask
        mask_filename = img_filename
        for i_s, m_s in zip(img_strs, mask_strs):
            mask_filename = mask_filename.replace(i_s, m_s)
        return mask_filename

    def _load_cache_file(self):
        path = Path(self.cache_path)
        train = path / 'train.txt'
        dev = path / 'val.txt'
        test = path / 'test.txt'
        if train.exists() and dev.exists() and test.exists():

            with train.open(mode='r', encoding='utf-8') as f:
                self.train_data_path = []
                for line in f:
                    self.train_data_path.append(line.strip())

            with dev.open(mode='r', encoding='utf-8') as f:
                self.dev_data_path = []
                for line in f:
                    self.dev_data_path.append(line.strip())

            with test.open(mode='r', encoding='utf-8') as f:
                self.test_data_path = []
                for line in f:
                    self.test_data_path.append(line.strip())
            return True
        else:
            return False

    def __repr__(self):
        return str(type(self).__name__) + " => train_net data : {}, dev data : {}, test data : {}".format(
            len(self.train_data_path), len(self.dev_data_path), len(self.test_data_path))


class ZeroLevelKFoldSplitter(BaseSplitter):
    """ Split images with kfolds
    input folder structure:
    Folder => images [only imgs]
    data_path : folder path
    """

    def __init__(self, data_path, img_to_mask, cache_path=None,
                 train_rate=0.7, dev_rate=0.3,
                 img_filter=None, not_image_filter=None,
                 force_cache=False, shuffle_seed=12345):
        self.imgs = list(Path(data_path).iterdir())
        self.imgs.sort()
        if shuffle_seed is not None:
            np.random.seed(shuffle_seed)
            np.random.shuffle(self.imgs)
        super(ZeroLevelKFoldSplitter, self).__init__(data_path, img_to_mask=img_to_mask,
                                                     train_rate=train_rate, dev_rate=dev_rate,
                                                     cache_path=cache_path, img_filter=img_filter,
                                                     not_img_filter=not_image_filter, force_cache=force_cache)

    def split_file(self):
        train_imgs, dev_imgs, test_imgs = split_trian_dev_test_path(self.imgs, train_rate=self.train_rate,
                                                                    dev_rate=self.dev_rate)
        self.add_file_to_data(train_imgs, self.train_data_path)
        self.add_file_to_data(dev_imgs, self.dev_data_path)
        self.add_file_to_data(test_imgs, self.test_data_path)


class OneLevelKFoldSplitter(BaseSplitter):
    """ Split cases with kfold
    input folder structure:
    Folder => case1 => imgs
              case2 => imgs
    data_path : folder path
    """

    def __init__(self, data_path, img_to_mask, cache_path=None,
                 train_rate=0.7, dev_rate=0.3,
                 img_filter=None, force_cache=False,
                 shuffle_seed=None):
        self.folders = list(filter(lambda x: x.is_dir(), list(Path(data_path).iterdir())))
        self.folders.sort()
        if shuffle_seed is not None:
            np.random.seed(shuffle_seed)
            np.random.shuffle(self.folders)
        super(OneLevelKFoldSplitter, self).__init__(data_path, img_to_mask=img_to_mask,
                                                    train_rate=train_rate, dev_rate=dev_rate,
                                                    cache_path=cache_path, img_filter=img_filter,
                                                    force_cache=force_cache)

    def split_file(self):
        train_folders, dev_folders, test_folders = split_trian_dev_test_path(self.folders, self.train_rate,
                                                                             self.dev_rate)
        for folder in train_folders:
            self.add_folder_to_data(folder, self.train_data_path)
        for folder in dev_folders:
            self.add_folder_to_data(folder, self.dev_data_path)
        for folder in test_folders:
            self.add_folder_to_data(folder, self.test_data_path)


class OneLevelSplitter(BaseSplitter):
    """If the folder only has one depth which means the structure is :
    Folder => train => imgs
           => dev   => imgs
           => test  => imgs
    """

    def __init__(self, data_path, img_to_mask, cache_path="./",
                 train_name='train', val_name='val', test_name='test',
                 img_filter='.jpg', force_cache=False):
        super(OneLevelSplitter, self).__init__(data_path=data_path, img_to_mask=img_to_mask,
                                               cache_path=cache_path,
                                               img_filter=img_filter, train_name=train_name,
                                               val_name=val_name, test_name=test_name, force_cache=force_cache)

    def split_file(self):
        root = Path(self.data_path)

        self.add_folder_to_data(root / self.train_name, self.train_data_path)
        self.add_folder_to_data(root / self.val_name, self.dev_data_path)
        self.add_folder_to_data(root / self.test_name, self.test_data_path)


class TwoLevelSplitter(BaseSplitter):
    """If the folder has two depth which means the structure is :
    Folder => train => cases => imgs
           => dev   => cases => imgs
           => test  => cases => imgs
    """

    def __init__(self, data_path, img_to_mask, cache_path="./",
                 train_name='train', val_name='val', test_name='test',
                 img_filter=None, force_cache=False):
        super(TwoLevelSplitter, self).__init__(data_path=data_path, train_name=train_name, img_to_mask=img_to_mask,
                                               val_name=val_name, test_name=test_name, img_filter=img_filter,
                                               cache_path=cache_path, force_cache=force_cache)

    def split_file(self):
        root = Path(self.data_path)

        for folder in (root / self.train_name).iterdir():
            self.add_folder_to_data(folder, self.train_data_path)

        for folder in (root / self.val_name).iterdir():
            self.add_folder_to_data(folder, self.dev_data_path)

        for folder in (root / self.test_name).iterdir():
            self.add_folder_to_data(folder, self.test_data_path)


class ThreeLevelCaseSplitter(BaseSplitter):
    """If the folder has two depth which means the structure is :
    Root   => train => Image/Mask => cases => imgs
           => dev   => Image/Mask => cases => imgs
           => test  => Image/Mask => cases => imgs
    """

    def __init__(self, data_path, img_to_mask,
                 train_name='train', val_name='val', test_name='test',
                 middle_folder='Image',
                 img_filter=None, cache_path=None, force_cache=False):
        self.middle_folder = middle_folder
        super(ThreeLevelCaseSplitter, self).__init__(data_path=data_path, train_name=train_name, img_to_mask=img_to_mask,
                                                 val_name=val_name, test_name=test_name, img_filter=img_filter,
                                                 cache_path=cache_path, force_cache=force_cache)

    def split_file(self):
        root = Path(self.data_path)
        for folder in (root / self.train_name / self.middle_folder).iterdir():
            self.add_folder_to_data(folder, self.train_data_path)

        for folder in (root / self.val_name / self.middle_folder).iterdir():
            self.add_folder_to_data(folder, self.dev_data_path)

        for folder in (root / self.test_name / self.middle_folder).iterdir():
            self.add_folder_to_data(folder, self.test_data_path)


class ThreeLevelSplitter(BaseSplitter):
    """If the folder has two depth which means the structure is :
    Root   => train => Image/Mask => imgs
           => dev   => Image/Mask => imgs
           => test  => Image/Mask => imgs
    """

    def __init__(self, data_path, img_to_mask,
                 train_name='train', val_name='val', test_name='test',
                 middle_folder='Image',
                 img_filter=None, cache_path=None, force_cache=False):
        self.middle_folder = middle_folder
        super(ThreeLevelSplitter, self).__init__(data_path=data_path, train_name=train_name, img_to_mask=img_to_mask,
                                                 val_name=val_name, test_name=test_name, img_filter=img_filter,
                                                 cache_path=cache_path, force_cache=force_cache)

    def split_file(self):
        root = Path(self.data_path)
        for folder in (root / self.train_name / self.middle_folder).iterdir():
            self.add_file_to_data(folder, self.train_data_path)

        for folder in (root / self.val_name / self.middle_folder).iterdir():
            self.add_file_to_data(folder, self.dev_data_path)

        for folder in (root / self.test_name / self.middle_folder).iterdir():
            self.add_file_to_data(folder, self.test_data_path)


class TrainTestOneLevelKFoldSplitter(BaseSplitter):
    """If the folder only has one depth which means the structure is :
    Folder => train => imgs
           => test  => imgs
    """

    def __init__(self, data_path, img_to_mask,
                 train_name='train', test_name='test',
                 train_rate=0.7, shuffle_seed=None,
                 img_filter=None, cache_path=None, force_cache=False):
        self.shuffle_seed=shuffle_seed
        super(TrainTestOneLevelKFoldSplitter, self).__init__(data_path=data_path, train_name=train_name, img_to_mask=img_to_mask,
                                                             test_name=test_name, img_filter=img_filter, train_rate=train_rate,
                                                             cache_path=cache_path, force_cache=force_cache)

    def split_file(self):
        root = Path(self.data_path)
        train_path = root / self.train_name
        test_path  = root / self.test_name

        train_folders = list(filter(lambda x: x.is_dir(), list(train_path.iterdir())))
        test_folders = list(filter(lambda x: x.is_dir(), list(test_path.iterdir())))

        train_folders.sort()
        if self.shuffle_seed is not None:
            np.random.seed(self.shuffle_seed)
            np.random.shuffle(train_folders)

        train_folders, dev_folders, _ = split_trian_dev_test_path(train_folders, self.train_rate, 1 - self.train_rate)

        for folder in train_folders:
            self.add_folder_to_data(folder, self.train_data_path)
        for folder in dev_folders:
            self.add_folder_to_data(folder, self.dev_data_path)
        for folder in test_folders:
            self.add_folder_to_data(folder, self.test_data_path)


class TrainTestZeroLevelKFoldSplitter(BaseSplitter):
    """If the folder only has one depth which means the structure is :
    Folder => train => imgs
           => test  => imgs
    """

    def __init__(self, data_path, img_to_mask,
                 train_name='train', test_name='test',
                 train_rate=0.7, shuffle_seed=None,
                 img_filter=None, cache_path=None, force_cache=False):
        self.shuffle_seed=shuffle_seed
        super(TrainTestZeroLevelKFoldSplitter, self).__init__(data_path=data_path, train_name=train_name, img_to_mask=img_to_mask,
                                                             test_name=test_name, img_filter=img_filter, train_rate=train_rate,
                                                             cache_path=cache_path, force_cache=force_cache)

    def split_file(self):
        root = Path(self.data_path)
        train_path = root / self.train_name
        test_path  = root / self.test_name

        train_files = list(train_path.iterdir())
        test_files =list(test_path.iterdir())

        train_files.sort()

        train_files, dev_files, _ = split_trian_dev_test_path(train_files, self.train_rate, 1 - self.train_rate)

        for file in train_files:
            self.add_file_to_data(file, self.train_data_path)
        for file in dev_files:
            self.add_file_to_data(file, self.dev_data_path)
        for file in test_files:
            self.add_file_to_data(file, self.test_data_path)


class CSVSplitter(BaseSplitter):
    def __init__(self, data_path, img_to_mask,
                 train_name='train.csv', val_name='test.csv', test_name='test.csv',
                 cache_path="./", img_filter='.jpg', force_cache=False):
        super(CSVSplitter, self).__init__(data_path=data_path, img_to_mask=img_to_mask,
                                          train_name=train_name, val_name=val_name, test_name=test_name,
                                          cache_path=cache_path, img_filter=img_filter, force_cache=force_cache)

    def split_file(self):
        with open(self.data_path / self.train_name, mode='r') as f:
            for line in f:
                self.train_data_path.append(line)

        with open(self.data_path / self.val_name, mode='r') as f:
            for line in f:
                self.dev_data_path.append(line)

        with open(self.data_path / self.test_name, mode='r') as f:
            for line in f:
                self.test_data_path.append(line)


class MatSplitter(BaseSplitter):
    def __init__(self, data_path, mat_path, mid_folder_name, img_to_mask,
                 train_name='trainIds', val_name='valIds', test_name='testIds',
                 cache_path="./", img_filter='.png', force_cache=False):
        self.mat_path = mat_path
        self.mid_folder_name = mid_folder_name
        super(MatSplitter, self).__init__(data_path=data_path, img_to_mask=img_to_mask,
                                          train_name=train_name, val_name=val_name, test_name=test_name,
                                          cache_path=cache_path, img_filter=img_filter, force_cache=force_cache)

    def split_file(self):
        import scipy.io as scio
        mat = scio.loadmat(str(self.mat_path))

        train_file = mat[self.train_name]
        train_file = train_file.squeeze()
        for id in train_file:
            path = self.data_path / self.mid_folder_name / '{:05d}.png'.format(id)
            self.add_file_to_data(str(path), self.train_data_path)

        val_file = mat[self.val_name]
        val_file = val_file.squeeze()
        for id in val_file:
            path = self.data_path / self.mid_folder_name / '{:05d}.png'.format(id)
            self.add_file_to_data(str(path), self.dev_data_path)

        test_file = mat[self.test_name]
        test_file = test_file.squeeze()
        for id in test_file:
            path = self.data_path / self.mid_folder_name / '{:05d}.png'.format(id)
            self.add_file_to_data(str(path), self.test_data_path)


if __name__ == '__main__':
    # root = '/home/zj/Medical_proj/data/patches3'
    # OneLevelKFoldSplitter(root, folds=4, img_to_mask=[['imaging'], ['segmentation']], force_cache=True)
    pass
