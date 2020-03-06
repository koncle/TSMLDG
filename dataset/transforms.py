import numpy as np
import numpy.random as random
import torch
from PIL import ImageFilter, Image as Img
from PIL.Image import BILINEAR, NEAREST, Image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from torchvision.transforms import functional as F

__all__ = ['Compose', 'Resize', 'RandomCrop', 'TargetCrop', 'RandomFlip',
           'random_gaussian_blur', 'RandomRotate', 'RandomScale',
           'ToTenser', 'medical_image_normalize', 'contrast_img_when_neccessary',
           'FixScale', 'CenterCrop', 'do_nothing', 'natural_image_normalize']


class Compose():
    """ Compose functions below
    :param functions:
    :return:
    Usage :
    >>> Compose([Resize([1, 2], BILINEAR), RandomFlip()])
    """
    def __init__(self, functions):
        self.funcs = functions

    def __call__(self, img, mask, mode):
        for func in self.funcs:
            if func is not None:
                img, mask = func(img, mask, mode)
        return img, mask


class Resize():
    """Resize PIL Image size:(H, W)
    :param size:
    :param interpolation:
    :return:
    """
    def __init__(self, size, mask_size=None):
        self.size = size
        self.mask_size = mask_size

    def __call__(self, img: Image, mask: Image, mode):
        img = F.resize(img, self.size, BILINEAR)
        if mask is not None and self.mask_size is not None:
            mask = F.resize(mask, self.mask_size, NEAREST)
        return img, mask


class RandomFlip():
    """Random flip PIL Image
    :param direction: ['h', 'v', 'both']
    :param show:
    :param rand:
    :return:
    """
    def __init__(self, rand=0.5, direction='h', show=False):
        self.rand = rand
        self.direction = direction

    def __call__(self, img: Image, mask: Image, mode):
        if (self.direction == 'h' or self.direction == 'both') and random.random() < self.rand:
            img = F.hflip(img)
            if mask is not None:
                mask = F.hflip(mask)
        if (self.direction == 'v' or self.direction == 'both') and random.random() < self.rand:
            img = F.vflip(img)
            if mask is not None:
                mask = F.vflip(mask)
        return img, mask



class RandomScale():
    """Random scale PIL Image
    :param scale_range:
    :return:
    """
    def __init__(self, base_long_size=None, scale_range=(0.75, 1.20)):
        self.base_long_size = base_long_size
        self.scale_range = scale_range

    def __call__(self, img: Image, mask: Image, mode):
        w, h = img.size
        if self.base_long_size is None:
            origin_size = h if w > h else w
        else:
            origin_size = self.base_long_size
        long_size = random.randint(int(origin_size * self.scale_range[0]), int(origin_size * self.scale_range[1]))

        if w < h:
            out_h = long_size
            ratio = out_h / h
            out_w = int(w * ratio)
        else:
            out_w = long_size
            ratio = out_w / w
            out_h = int(h * ratio)

        img = F.resize(img, (out_h, out_w), interpolation=BILINEAR)
        if mask is not None:
            mask = F.resize(mask, (out_h, out_w), interpolation=NEAREST)
        return img, mask


class FixScale():
    """Random scale PIL Image
    :param scale_range:
    :return:
    """
    def __init__(self, short_size):
        self.short_size = short_size

    def __call__(self, img: Image, mask: Image, mode):
        w, h = img.size
        if w > h:
            out_h = self.short_size
            ratio = (1.0 * out_h) / h
            out_w = int(w * ratio)
        else:
            out_w = self.short_size
            ratio = (1.0 * out_w) / w
            out_h = int(h * ratio)

        img = F.resize(img, (out_h, out_w), interpolation=BILINEAR)
        if mask is not None:
            mask = F.resize(mask, (out_h, out_w), interpolation=NEAREST)
        return img, mask


class RandomRotate():
    """Random rotate PIL Image
    :param angle_range:
    :return:
    """
    def __init__(self, angle_range=(-10, 10), rand=0.5):
        self.angle_range = angle_range
        self.rand = rand

    def __call__(self, img: Image, mask: Image, mode):
        if random.random() < self.rand:
            rotate_angle = random.uniform(*self.angle_range)
            img = F.rotate(img, rotate_angle, resample=BILINEAR)
            if mask is not None:
                mask = F.rotate(mask, rotate_angle, resample=NEAREST)
                # mask = F.rotate(mask, rotate_angle, resample=NEAREST, fill=(0,))
        return img, mask


class CenterCrop():
    def __init__(self, size=[256, 256], img_fill=0, mask_fill=0, test_crop=False):
        self.size = size
        self.img_fill = img_fill
        self.mask_fill = mask_fill

    def __call__(self, img: Image, mask: Image, mode):
        w, h = img.size
        # Padding img before crop if the size of img is to small
        if w <= self.size[0] or w <= self.size[1]:
            pad_w = self.size[0] - w if w < self.size[0] else 0
            pad_h = self.size[0] - h if h < self.size[1] else 0
            # left, top, right, bottom
            img = F.pad(img, (0, 0, pad_w, pad_h), fill=self.img_fill)
            if mask is not None:
                mask = F.pad(mask, (0, 0, pad_w, pad_h), fill=self.mask_fill)
        w, h = img.size
        if w == self.size[0] and h == self.size[1]:
            return img, mask

        if h == self.size[1]:
            i = 0
        else:
            i = (h - self.size[1]) // 2

        if w == self.size[0]:
            j = 0
        else:
            j = (w - self.size[0]) // 2

        img = F.crop(img, i, j, self.size[0], self.size[1])
        if mask is not None:
            mask = F.crop(mask, i, j, self.size[0], self.size[1])
        return img, mask


class TargetCrop():
    """
    Crop the image at where the target(mask > 0) centers.
    Random crop when no target in the image.
    WARN : min_size should be larget than the target size.
    :param min_size:
    :param img_fill:
    :param mask_fill:
    :return:
    """
    def __init__(self, min_size = [256, 256], img_fill = 0, mask_fill = 0):
        self.min_size = min_size
        self.img_fill = img_fill
        self.mask_fill = mask_fill
        self.rand_crop = RandomCrop(size=min_size, img_fill=img_fill, mask_fill=mask_fill)

    def __call__(self, img: Image, mask: Image, mode):
        from preprocess.generate_bbox import get_bbox_2d
        assert mask is not None
        mask_data = np.array(mask)
        img_data = np.array(img)
        img_y, img_x = mask_data.shape

        (min_x, max_x), (min_y, max_y) = get_bbox_2d(mask_data)

        if (max_x - min_x) == 1:
            (min_x, max_x), (min_y, max_y) = get_bbox_2d(img_data)
            if (max_x - min_x) == 1:
                return self.rand_crop(img, mask, mode)

        bbox_h = max_y - min_y
        bbox_w = max_x - min_x

        # pad bounding box
        pad_h = np.max((self.min_size[0]-bbox_h, 0))
        pad_w = np.max((self.min_size[1]-bbox_w, 0))

        i = np.max((0, min_y-pad_h//2))
        j = np.max((0, min_x-pad_w//2))

        bbox_h = np.min((i + self.min_size[0], img_y)) - i
        bbox_w = np.min((j + self.min_size[1], img_x)) - j

        img  = F.crop(img, i, j, bbox_h, bbox_w)
        mask = F.crop(mask, i, j, bbox_h, bbox_w)

        # pad cropped image to at least min_size
        pad_h = np.max((self.min_size[0]-bbox_h, 0))
        pad_w = np.max((self.min_size[1]-bbox_w, 0))

        img = F.pad(img, (0, 0, pad_w, pad_h), fill=self.img_fill)
        mask = F.pad(mask, (0, 0, pad_w, pad_h), fill=self.mask_fill)
        return img, mask


class RandomCrop():
    """Random crop PIL Image, it will pad the image first if the size is not enough
    :param size:
    :return:
    """
    def __init__(self, size=[256, 256], img_fill=0, mask_fill=0, test_crop=False, show=False):
        self.size = size
        self.img_fill = img_fill
        self.mask_fill = mask_fill

    def __call__(self, img: Image, mask: Image, mode):
        w, h = img.size
        # Padding img before crop if the size of img is to small
        if w <= self.size[0] or h <= self.size[1]:
            pad_w = self.size[0] - w if w < self.size[0] else 0
            pad_h = self.size[1] - h if h < self.size[1] else 0
            # left, top, right, bottom
            img = F.pad(img, (0, 0, pad_w, pad_h), fill=self.img_fill)
            if mask is not None:
                mask = F.pad(mask, (0, 0, pad_w, pad_h), fill=self.mask_fill)
        w, h = img.size

        if w == self.size[0] and h == self.size[1]:
            return img, mask

        if h == self.size[1]:
            i = 0
        else:
            i = random.randint(0, h - self.size[1])

        if w == self.size[0]:
            j = 0
        else:
            j = random.randint(0, w - self.size[0])

        img = F.crop(img, i, j, self.size[0], self.size[1])
        if mask is not None:
            mask = F.crop(mask, i, j, self.size[0], self.size[1])
        return img, mask


class random_gaussian_blur():
    """Random gaussian blur with the possibility of rand
    :param rand: the possibility to blur image
    :return:
    """
    def __init__(self, rand=0.5):
        self.rand = rand

    def __call__(self, img: Image, mask: Image, mode):
        if random.random() < self.rand:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img, mask


# need to finetune gamma for cases
def gamma_correction(gamma=0.4):
    def gamma_correction(img: Image, mask: Image, mode):
        img = F.adjust_gamma(img, gamma=gamma)
        return img, mask

    return gamma_correction


class ToTenser():
    """ Convert PIL Image to torch tensor
    :return:
    """
    def __init__(self, img_type=torch.float32, mask_type=torch.long):
        self.img_type = img_type
        self.mask_type = mask_type

    def __call__(self, img, mask, mode):
        # image will be divided by 255. to range in [0., 1.]
        img1 = F.to_tensor(img).type(self.img_type)
        if mask is not None:
            mask = np.array(mask)
            mask = np.expand_dims(mask, 0)
            # mask will not be changed.
            mask = torch.from_numpy(mask).type(self.mask_type)
        return img1, mask


class natural_image_normalize():
    def __call__(self, img, mask, mode):
        img = F.normalize(img, [.485, .456, .406], [.229, .224, .225])
        return img, mask


def medical_image_normalize():
    """ Normalize **tensor** image!
    :return:
    """

    def normalize_img(img, mask, mode):
        eps = 1e-10
        # img = (img - img.mean() + eps) / (img.std() + eps)
        # img = (img - img.min() + eps) / (img.max() - img.min() + eps)
        # eps = 1e-10
        # channels = img.size()[0]
        # if channels == 1:
        #     img = (img - img.min() + eps) / (img.max() - img.min() + eps)
        # else:
        #     # channel wise normalize
        #     for channel in range(channels):
        #         img[channel, :] = (img[channel, :] - img[channel, :].min()) / \
        #                           (img[channel, :].max() - img[channel, :].min())
        return img, mask

    return normalize_img


def elastic_transform(alpha, sigma):
    def do_elastic_transfrom(img, mask, mode):
        img = np.array(img)
        if mask is not None:
            mask = np.array(mask)

        shape = img.shape
        dx = gaussian_filter((np.random.uniform(0, 1, shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.uniform(0, 1, shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

        distored_image = map_coordinates(img, indices, order=1, mode='reflect')
        img = distored_image.reshape(img.shape)

        if mask is not None:
            distored_image = map_coordinates(mask, indices, order=1, mode='reflect')
            mask = distored_image.reshape(mask.shape)
        return img, mask

    return do_elastic_transfrom


def do_nothing():
    def do_do_nothing(img, mask, mode):
        return img, mask
    return do_do_nothing


from skimage import exposure


def contrast_img_when_neccessary():
    def contrast_img(img:Image, mask:Image, mode):
        img = np.array(img)
        if exposure.is_low_contrast(img):
            img = exposure.adjust_gamma(img, 0.4)
            print(True)
        else:
            print(False)
        return Img.fromarray(img), mask
    return contrast_img


def random_flip_numpy():
    def random_flip_3d(img: Image, mask: Image, mode=None):
        if np.random.rand() < 0.5:
            img = img[::-1, ::-1, ...]
            if mask is not None:
                mask = mask[::-1, ::-1, ...]
        return img, mask

    return random_flip_3d
