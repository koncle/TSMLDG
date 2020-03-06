import warnings
from pathlib import Path

from PIL import Image
from torch.utils.data.dataset import Dataset

from dataset.transforms import ToTenser, medical_image_normalize, natural_image_normalize
from utils.nn_utils import mkdir

warnings.filterwarnings('ignore')


class BaseDataSet(Dataset):
    """
    transform : a function to be applied to img and mask
    """

    def __init__(self, root, output_path, force_cache, transform=None, mode='train', medical=False):
        super(BaseDataSet, self).__init__()
        self.root = Path(root)
        self.output_path = Path(output_path)
        mkdir(self.output_path, create_self=True)
        self.force_cache = force_cache
        self.transforms = transform
        self.mode = mode.lower()
        assert self.mode in ['train', 'test', 'val']

        self.medical = medical
        self.normalize = medical_image_normalize() if self.medical else natural_image_normalize()

        self.paths = self._split_data(root)

    def split_data(self, root):
        raise NotImplementedError()

    def _split_data(self, root):
        train_data, val_data, test_data = self.split_data(root)

        if self.mode == 'train':
            paths = train_data
        elif self.mode == 'test':
            paths = test_data
        elif self.mode == 'val':
            paths = val_data
        elif self.mode == 'full':
            paths = train_data + val_data + test_data
        return paths

    def load_img(self, img_filename, mask_filename, mode):
        """
        Load PIL Image from filename
        The output must be **PIL object** !!
        :param img_filename:
        :param is_img:
        :return:
        """
        img = Image.open(img_filename)
        if self.medical:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        try:
            mask = Image.open(mask_filename)
            mask = mask.convert('L')
        except Exception as e:
            mask = None
        return img, mask

    def post_process(self, img, mask, mode):
        """ process tensor image and mask after completing transformations
        :param mode:
        :param img:
        :param mask:
        :return:
        """
        return img, mask

    def get_img_path(self, index):
        return self.paths[index].split('\t')

    def __getitem__(self, index):
        img_filename, mask_filename = self.get_img_path(index)
        # TODO : remove to_tensor() and normalize()
        funcs = [self.load_img, self.transforms, ToTenser(), self.normalize, self.post_process]

        img = img_filename
        mask = mask_filename
        for f in funcs:
            if f is None:
                continue
            img, mask = f(img, mask, self.mode)
        return img_filename, img, mask

    def __len__(self):
        return len(self.paths)


class CombinedDataSet(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.total_length = 0
        self.intervals = [0]
        for dataset in self.datasets:
            self.total_length += len(dataset)
            self.intervals.append(self.total_length)

    def __getitem__(self, idx):
        for dataset_idx, interval in enumerate(self.intervals):
            if idx < interval:
                target_dataset = self.datasets[dataset_idx]
        return target_dataset[idx]

    def __len__(self):
        return self.total_length
