import numpy as np
import torch
from torch.nn import functional as F

from utils.nn_utils import to_one_hot, get_probability, make_same_size, to_numpy
import scipy.spatial.distance as dist
import skimage.segmentation as skseg
from scipy.ndimage import measurements

# __all__ = ['IOU', 'DICE', 'mean_iou', 'mean_dice', 'get_confusion_matrix', 'hausdorff_distance',
#            'average_surface_distance', 'get_inter_union', 'centroid_distance']


def get_confusion_matrix_gpu(prediction, target, nclass):
    """ Compute confusion matrix for prediction size=(nclass, nclass)
    :param prediction: (N, H, W) tensor
    :param target:     (N, H, W) tensor
    :param nclass:
    :return:
    """
    assert prediction.shape == target.shape, \
        "Shape mismatch pred.shape={}, target.shape={}".format(prediction.shape, target.shape)
    mask = (target >= 0) & (target < nclass)
    label = nclass * target[mask].long() + prediction[mask]
    count = torch.bincount(label, minlength=nclass ** 2)
    confusion_matrix = count.reshape(nclass, nclass)
    return confusion_matrix


def get_confusion_matrix(prediction, target, nclass):
    """ Compute confusion matrix for prediction size=(nclass, nclass)
    :param prediction: (N, H, W) tensor
    :param target:     (N, H, W) tensor
    :param nclass:
    :return:
    """
    prediction, target = to_numpy((prediction, target))
    assert prediction.shape == target.shape, \
        "Shape mismatch pred.shape={}, target.shape={}".format(prediction.shape, target.shape)
    mask = (target >= 0) & (target < nclass)
    label = nclass * target[mask].astype('int') + prediction[mask]
    count = np.bincount(label, minlength=nclass ** 2)
    confusion_matrix = count.reshape(nclass, nclass)
    return confusion_matrix


def pixel_acc(confusion_matrix:np.ndarray):
    return np.diag(confusion_matrix).sum() / confusion_matrix.sum()


def pixel_acc_per_class(confusion_matrix:np.ndarray):
    acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    return np.nanmean(acc)


def mIoU(confusion_matrix:np.ndarray):
    inter = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - inter
    iou = inter / union
    return np.nanmean(iou)


def FWIoU(confusion_matrix:np.ndarray):
    """ Frequency weighted mean IoU
    :param confusion_matrix:
    :return:
    """
    iou = mIoU(confusion_matrix)
    freq = confusion_matrix.sum(axis=1) / confusion_matrix.sum()
    FWIoU = (freq[freq>0]*iou[freq>0]).sum()
    return FWIoU


def hausdorff_distance(prediction:np.ndarray, target:np.ndarray):
    """ Compute Hausdorff distance. Takes about 0.7min for 422*6 images
    :param prediction: shape=(N, H, W)
    :param target:     shape=(N, H, W)
    :return:
    """
    assert prediction.shape == target.shape, \
        "Shape mismatch pred.shape={}, target.shape={}".format(prediction.shape, target.shape)
    assert 0 <= prediction.max() <= 1
    hd_list = []
    N, H, W = prediction.shape
    for pred_slice, target_slice in zip(prediction, target):
        pred_slice = skseg.find_boundaries(pred_slice, connectivity=1, mode='thick', background=0).astype(np.float32)
        target_slice = skseg.find_boundaries(target_slice, connectivity=1, mode='thick', background=0).astype(np.float32)

        pred_coord = np.array(np.where(pred_slice == 1)).transpose(1, 0)
        target_coord = np.array(np.where(target_slice == 1)).transpose(1, 0)
        if len(pred_coord) == 0 and len(target_coord) == 0:
            hd_list.append(0)
        elif len(pred_coord) != 0 and len(target_coord) != 0:
            hd_list.append(dist.directed_hausdorff(pred_coord, target_coord)[0])
        else:
            hd_list.append(0)
            # print('empty prediction cause {} distance'.format((H*W)**2))
    return np.nanmean(hd_list)


def get_inter_union(logits, target, ignore_idx=-1, should_sigmoid=True):
    """
    :param logits: N x C x H x W
    :param target: N x 1 x H x W
    :return:
    """
    N, C, H, W = logits.size()

    logits = logits[target != ignore_idx].view(N, C, -1)
    target = target[target != ignore_idx].view(N, 1, -1)

    size = list(logits.size())
    N, nclass = size[0], size[1]
    # N x 1 x H x W
    if should_sigmoid:
        if logits.size()[1] > 1:
            size[1] = 1
            pred = torch.argmax(F.softmax(logits, dim=1), dim=1).view(*size)
        else:
            pred = torch.round((F.sigmoid(logits))).type(torch.long)
            nclass = 2
    # N x C x H x W
    pred_one_hot = to_one_hot(pred, nclass)
    target_one_hot = to_one_hot(target.type(torch.long), nclass)

    inter = pred_one_hot * target_one_hot
    union = pred_one_hot + target_one_hot - inter

    # N x C
    inter = inter.view(N, nclass, -1).sum(2)
    union = union.view(N, nclass, -1).sum(2)

    return inter, union


# iou = (class_inter + self.eps) / (class_union + self.eps)
def IOU(inter, union):
    # size = [class,]
    eps = np.spacing(1)
    res = (inter + eps) / (union + eps)
    # mark : background is not included
    return res[:, 1:, ...].mean()


# dice = (2 * class_inter + self.eps) / (class_union + class_inter)
def DICE(inter, union):
    # size = [class,]
    eps = np.spacing(1)
    res = (2 * inter + eps) / (union + inter + eps)
    # mark : background is not included
    return res[:, 1:, ...].mean()


def mean_iou(logits, target, ignore_idx):
    """
    mean IOU
    :param logits: NxCxHxW
    :param target: Nx1xHxW
    :return:
    """
    # N x C
    inter, union = get_inter_union(logits, target, ignore_idx)
    return IOU(inter, union)


def mean_dice(logits, target, ignore_idx):
    """
    mean dice
    :param logits: N x C x H x W
    :param target: N x 1 x H x W
    :return:
    """
    # N x C
    inter, union = get_inter_union(logits, target, ignore_idx)
    return DICE(inter, union)


def cosine_similarity(logits, target):
    target = target.float()
    logits = make_same_size(logits, target)
    sim = (logits * target).sum() / ((logits ** 2).sum().sqrt() * (target ** 2).sum().sqrt())
    return (sim + 1) / 2


def mean_square_similarity(logits, target):
    logits = make_same_size(logits, target)
    target = target.float()
    return 1 - ((logits - target) ** 2).mean()


def SEN(logits, target):
    logits = make_same_size(logits, target)
    eps = np.spacing(1)
    # NxC
    inter, union = get_inter_union(logits, target)
    # NxC
    N, C, W, H = target.size()
    target = target.view(N, C, -1).sum(2)
    # NxC
    sen = ((inter + eps) / (target + eps))
    # remove background
    return sen[:, 1:].mean()


def PPV(logits, target):
    logits = make_same_size(logits, target)
    eps = np.spacing(1)
    # NxC
    pred, _ = get_probability(logits)
    # NxC
    inter, union = get_inter_union(logits, target)

    N, C, W, H = pred.size()
    pred = pred.view(N, C, -1).sum(2)

    # NxC
    ppv = ((inter + eps) / (pred + eps))
    # remove background
    return ppv[:, 1:].mean()

