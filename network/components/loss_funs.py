import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils_.nn_utils import make_same_size, to_one_hot, get_probability
import scipy.ndimage as nd


class CrossEntropy(nn.Module):
    def __init__(self, nclass=1, weight=None, ignore_idx=-1):
        super(CrossEntropy, self).__init__()
        if weight is not None:
            weight = torch.tensor(weight, requires_grad=False).type(torch.float32)
        self.bc = nn.CrossEntropyLoss(ignore_index=ignore_idx, weight=weight)

    def forward(self, logits, target):
        if isinstance(logits, (list, tuple)):
            assert len(logits) == 1
            logits = logits[0]
        logits = make_same_size(logits, target)
        target = target[:, 0, ...].long()
        return self.bc(logits, target)


class BCELoss(nn.Module):
    def __init__(self, nclass, weight=None):
        super(BCELoss, self).__init__()
        if weight is not None:
            weight = torch.tensor(weight, requires_grad=False).type(torch.float32)
        self._bce_loss = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, logits, target):
        logits_flatten = logits.view(-1).type(torch.float32)
        labels_flatten = target.view(-1).type(torch.float32)
        return self._bce_loss(logits_flatten, labels_flatten)


class WeightedCrossEntropy(nn.Module):
    def __init__(self, nclass, loss_weight=None, weight=None, reduction='mean', ignore_idx=-1):
        super(WeightedCrossEntropy, self).__init__()
        if loss_weight is not None:
            if ignore_idx >= 0:
                assert ignore_idx < len(loss_weight)
                loss_weight[ignore_idx] = 0
            self.loss_weight = loss_weight
            if isinstance(self.loss_weight, list):
                self.loss_weight = nn.Parameter(torch.tensor(loss_weight, requires_grad=False).type(torch.float32))
        else:
            self.loss_weight = None
        if weight is not None:
            self.weight = nn.Parameter(torch.tensor(loss_weight, requires_grad=False).type(torch.float32))
        else:
            self.weight = None
        self.reduction = reduction
        self.ignore_idx = ignore_idx

    def forward(self, logits, target):
        if logits.size() != target.size():
            logits = make_same_size(logits, target)
        target = target.long()
        dims = logits.size()
        if len(dims) == 2:
            # 1d
            pass
        elif len(dims) == 4:
            # 2d
            n, c, w, h = dims
            if self.loss_weight is not None and len(self.loss_weight) != c:
                raise Exception("Expected {} channels, but got {} for size {}. current loss_weight {}."
                                .format(c, len(self.loss_weight), dims, self.loss_weight))

            # put the channel the last position to get correct result
            logits = logits.permute([0, 2, 3, 1]).contiguous().view(-1, c)
            target = target.view(-1)
        else:
            # wrong
            raise Exception("Wrong type")

        # negtive log likelihood
        neg_log_likelihood = -F.log_softmax(logits, 1)

        # one hot of target
        one_hot = torch.zeros_like(neg_log_likelihood)
        one_hot_target = one_hot.scatter_(1, target.reshape(-1, 1), 1)

        # weight for imbalanced classes
        if self.weight is not None:
            loss_ = (one_hot_target * neg_log_likelihood * self.weight).sum(dim=1)
        else:
            loss_ = (one_hot_target * neg_log_likelihood).sum(dim=1)

        # loss weight
        if self.loss_weight is not None:
            if len(self.loss_weight.size()) > 1:
                weight_map = self.get_2d_cost_weight_map(logits, target)
            else:
                weight_map = self.get_1d_cost_weight_map(logits, target)
            loss_ *= weight_map

        if self.reduction == 'mean':
            return loss_.mean()
        elif self.reduction == 'sum':
            return loss_.sum()
        else:
            if len(dims) > 2:
                return loss_.view((n, w, h))

    def get_1d_cost_weight_map(self, logits, target):
        # get weight for every different prediction
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        # 0-1 weight
        weight_map = (target != pred).type(torch.float32)
        # assign weight to 1
        for class_, w in enumerate(self.loss_weight):
            # find elements that equals class
            weight_map[class_ == target] *= w
        # convert 0 to 1 to preserve correct loss
        weight_map[weight_map == 0] = 1
        return weight_map

    def get_2d_cost_weight_map(self, logits, target):
        # get weight for every different prediction
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        weight_map = (target != pred).type(torch.float32)
        t = target.type(torch.long)
        p = target.type(torch.long)
        for class_ in range(self.loss_weight.size()[0]):
            # find elements that equals class
            idx = class_ == target
            weight_map[idx] *= self.loss_weight[t[idx], p[idx]]
        weight_map[weight_map == 0] = 1
        return weight_map


class DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def forward(self, logits, target):
        logits = make_same_size(logits, target)

        size = logits.size()
        N, nclass = size[0], size[1]

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        # N x C
        inter = inter.view(N, nclass, -1).sum(2)
        union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        weighted_dice = dice * self.class_weights
        # sum of all class, mean of all batch
        #
        # use which one ? weighted_dice.mean(), weighted_dice.sum(1).mean(), weighted_dice.sum(0).mean()
        return 1 - weighted_dice.mean()


class BoundedDice(nn.Module):
    def __init__(self, nclass):
        super(BoundedDice, self).__init__()
        self.dice = DiceLoss(nclass)

    def forward(self, logits_list, target_list):
        total_loss = 0
        num_loss = 0
        for i in range(len(logits_list)):
            logits = logits_list[i]
            target, bound = target_list[1][i]
            (min_x, min_y, min_z, max_x, max_y, max_z) = bound
            if min_x == max_x:
                continue
            target = target[None, :, min_z:max_z, min_y:max_y, min_x, max_x]
            loss = self.dice(logits, target)
            total_loss += loss
            num_loss += 1
        return total_loss / num_loss


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, nclass):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, logits, target):
        logits = make_same_size(logits, target)

        size = logits.size()
        N, nclass = size[0], size[1]

        prob, nclass = get_probability(logits)

        # N x C x H x W
        prob_one_hot = prob
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = prob_one_hot * target_one_hot
        union = prob_one_hot ** 2 + target_one_hot ** 2

        # N x C
        inter = inter.view(N, nclass, -1).sum(2).sum(0)
        union = union.view(N, nclass, -1).sum(2).sum(0)

        # NxC
        dice = 2 * inter / union
        return dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, nclass, gamma=2, alpha=0.5):
        self.gamma = gamma
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                             size_average=self.size_average)

    def forward(self, logit, target):
        n, c, h, w = logit.size()
        logpt = -self.criterion(logit, target.long())
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


class SimplifiedDice(nn.Module):
    def __init__(self, nclass, smooth=1e-5):
        super(SimplifiedDice, self).__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        logits = make_same_size(logits, target)
        target = target.float()
        size = logits.size()
        assert size[1] == 1  # only support binary case
        logits = logits.view(size[0], -1)
        target = target.view(size[0], -1)
        inter = (logits * target).sum(1)
        union = (logits + target).sum(1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return -dice.mean()


class WeightedMSE(nn.Module):
    def __init__(self, nclass):
        super(WeightedMSE, self).__init__()

    def forward(self, logits, target):
        target = target.float()
        pos_loss = ((logits[target > 0] - target[target > 0]) ** 2) / (target > 0).sum().float()
        neg_loss = ((logits[target <= 0] - target[target <= 0]) ** 2) / (target <= 0).sum().float()
        return (pos_loss.sum() + neg_loss.sum()) / 2 * logits.size()[0]


class EmptyLoss(nn.Module):
    def __init__(self, nclass):
        super(EmptyLoss, self).__init__()

    def forward(self, logits, target):
        tmp = torch.Tensor([0])
        tmp.requires_grad = True
        return tmp.sum()


class OhemCrossEntropy2d(nn.Module):

    def __init__(self, nclass, ignore_label=-1, thresh=0.6, min_kept=200000, factor=8):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        # self.min_kept_ratio = float(min_kept_ratio)
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_label)

    def find_threshold(self, np_predict, np_target):
        # downsample 1/8
        factor = self.factor
        predict = nd.zoom(np_predict, (1.0, 1.0, 1.0 / factor, 1.0 / factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0 / factor, 1.0 / factor), order=0)

        n, c, h, w = predict.shape
        min_kept = self.min_kept // (factor * factor)  # int(self.min_kept_ratio * n * h * w)

        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        # if valid number is smaller than num that need to be kept
        if min_kept >= num_valid:
            threshold = 1.0

        # else keep
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_array = np.partition(pred, k_th)
                new_threshold = new_array[k_th]
                if new_threshold > self.thresh:
                    threshold = new_threshold
        return threshold

    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        n, c, h, w = np_predict.shape

        threshold = self.find_threshold(np_predict, np_target)

        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, 1).reshape((c, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()

        if num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            print('Labels: {} {}'.format(len(valid_inds), threshold))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(target.get_device())

        return new_target

    def forward(self, logits, target, weight=None):
        """
            Args:
                logits:(n, c, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        target = target[:, 0]
        logits = make_same_size(logits, target)
        input_prob = F.softmax(logits, 1)
        target = self.generate_new_target(input_prob, target)
        return self.criterion(logits, target)
