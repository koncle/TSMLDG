from engine import configs
from network.components.loss_funs import *


"""
For log sake...
Note1 : loss can only be scala type not tensor.
Note2 : Try to pass param to wrapper, instead of inheriting it 
            so that your loss function can be used in other places.
"""


def tensor_to_scala(data):
    if isinstance(data, list) or isinstance(data, tuple):
        new_data = []
        for l in data:
            if isinstance(l, torch.Tensor):
                new_data.append(l.item())
            else:
                new_data.append(l)
    elif isinstance(data, torch.Tensor):
        new_data = data.item()
    else:
        new_data = data
    return new_data


class LoggedLoss(nn.Module):
    def __init__(self, names=[], max_length=5000):
        super(LoggedLoss, self).__init__()
        self.cached_total_loss = []
        self.cached_split_loss = []
        self.names = names
        self.max_length = max_length

    def forward(self, logits, target):
        raise NotImplementedError()

    def append_loss(self, total_loss, split_loss):
        """ total_loss and split_loss should be float or int , not tensor
        :param total_loss: float
        :param split_loss: list of float
        :return:
        """
        total_loss = tensor_to_scala(total_loss)
        split_loss = tensor_to_scala(split_loss)

        # assert not isinstance(total_loss, torch.Tensor),
        # "total_loss and split_loss should be float or int , not tensor"

        if isinstance(split_loss, list) or isinstance(split_loss, tuple):
            self.cached_split_loss.append(split_loss)
        else:
            self.cached_split_loss.append([split_loss])

        self.cached_total_loss.append(total_loss)
        if len(self.cached_total_loss) == self.max_length:
            self.cached_total_loss.pop(0)

    def get_split_loss(self):
        return list(np.mean(self.cached_split_loss, 0))

    def get_loss(self):
        return np.mean(self.cached_total_loss)

    def get_last(self):
        return self.cached_total_loss[-1]

    def get_last_split(self):
        return list(self.cached_split_loss[-1])

    def get_loss_length(self):
        return len(self.names)

    def clear_cache(self):
        self.cached_total_loss = []
        self.cached_split_loss = []

    def get_loss_names(self):
        return self.names


class SingleLossWrapper(LoggedLoss):
    """ A wrapper for single loss function to record loss.

        To help user know how many loss get for an epoch,
        this wrapper is created to hold the origin loss function
        and record its output loss simultaneously.
    """
    def __init__(self, loss_func=None):
        super(SingleLossWrapper, self).__init__()
        self.loss_func = loss_func

    def forward(self, logits, target):
        l = self._calculate_loss(logits, target)

        if isinstance(l, (tuple, list)):
            raise RuntimeError("A list of loss is returned which should be implemented with MultiLossWrapper")

        if l is None:
            raise RuntimeError("A loss should be returned")

        self.append_loss(l, l)
        return l

    def _calculate_loss(self, logits, target):
        if self.loss_func is not None:
            return self.loss_func(logits, target)
        else:
            raise Exception("Override this method or pass loss_func param into the class")


class MultiLossWrapper(LoggedLoss):
    """ A wrapper for multi loss function to record multi losses.

      Sometimes, a network will return many outputs, then use different
      loss functions to compute final loss. Thus, this class putes
      these loss functions together to generate a single output.

      Also it records loss history to print to console.
    :param
      loss_funcs : a loss function list, [func1, func2...]
      weights    : a list of weights assigned for corresponding loss
      names      : loss function names, default is [loss_1, loss_2, ...]
    """
    def __init__(self, names=[], loss_funcs=None):
        if len(names) == 0:
            raise Exception('names should be specified for each loss')
        super(MultiLossWrapper, self).__init__(names=names)
        self.loss_funcs = loss_funcs

    def _calculate_loss(self, logits, target):
        if self.loss_funcs is not None:
            return self.loss_funcs(logits, target)
        else:
            raise Exception("Override this method or pass loss_func param into the class to log losses")

    def forward(self, logits, target):
        losses = self._calculate_loss(logits, target)
        if not isinstance(losses, (tuple, list)):
            losses = [losses]

        assert len(losses) == len(self.names), \
            "the number of loss {} should be the same as names' {}".format(len(losses), len(self.names))

        total_loss = 0
        for loss in losses:
            total_loss += loss

        self.append_loss(total_loss, losses)
        return total_loss


class SameTargetLossWrapper(MultiLossWrapper):
    def __init__(self, loss_num, loss_func):
        names = ['l'+str(i) for i in range(loss_num)]
        super(SameTargetLossWrapper, self).__init__(names=names, loss_funcs=loss_func)

    def _calculate_loss(self, logits, target):
        assert isinstance(logits, (tuple, list)) and \
               (not isinstance(target, (tuple, list)) or len(logits) == len(target))

        losses = []
        if isinstance(target, (tuple, list)):
            for l, t in zip(logits, target):
                losses.append(self.loss_funcs(l, t))
        else:
            for l in logits:
                losses.append(self.loss_funcs(l, target))
        return losses


@configs.LossFuncs.register('seg')
class SegmentationLoss(MultiLossWrapper):
    def __init__(self, nclass, aux=True, aux_weight=0.4, ohem=False, ignore_idx=-1, loss='ce'):
        self.aux_weight = aux_weight
        names = ['seg']

        names.append('aux')
        if loss == 'ce':
            loss = CrossEntropy
        elif loss == 'part':
            loss = SoftTriple

        super(SegmentationLoss, self).__init__(names=names)

        if ohem:
            self.loss_1 = OhemCrossEntropy2d(nclass, ignore_label=ignore_idx, thresh=0.6, min_kept=200000, factor=8)
        else:
            self.loss_1 = loss(nclass, ignore_idx=ignore_idx)

        self.loss_2 = CrossEntropy(nclass, ignore_idx=ignore_idx)

        self.true_prob_list = []
        self.false_prob_list = []

    def _calculate_loss(self, logits, target):
        assert isinstance(logits, (tuple, list)), 'output must be tupel or list'
        logits[0] = F.interpolate(logits[0], size=target.size()[2:], mode='bilinear', align_corners=True)
        seg_loss = self.loss_1(logits[0], target)
        losses = [seg_loss]
        aux = 0
        for logit in logits[1:]:
            aux_loss = self.loss_2(logit, target) * self.aux_weight
            aux += aux_loss
        losses.append(aux)
        return losses


@configs.LossFuncs.register('direct')
class DirectLoss(SingleLossWrapper):
    def __init__(self):
        super(DirectLoss, self).__init__()

    def _calculate_loss(self, logits, target):
        return logits

# Implementation of SoftTriple Loss
import math
from torch.nn.parameter import Parameter
from torch.nn import init


class SoftTriple(nn.Module):
    def __init__(self, cN=5, K=19, dim=19, lambda_=1, gamma=0.1, tau=0.2, margin=0.01, ignore_idx=-1):
        super(SoftTriple, self).__init__()
        self.la = lambda_
        self.gamma = 1./gamma
        self.tau = tau
        self.margin = margin
        self.cN = cN
        self.K = K
        self.ignore_idx = ignore_idx
        self.fc = nn.Parameter(torch.Tensor(dim, cN*K))
        self.weight = torch.zeros(cN*K, cN*K, dtype=torch.bool).cuda()
        for i in range(0, cN):
            for j in range(0, K):
                self.weight[i*K+j, i*K+j+1:(i+1)*K] = 1
        nn.init.kaiming_uniform_(self.fc, a=math.sqrt(5))
        return

    def forward(self, input, target):
        N, C, H, W = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(N*H*W, C)

        centers = F.normalize(self.fc, p=2, dim=0)
        simInd = input.matmul(centers)
        simStruc = simInd.reshape(-1, self.cN, self.K)
        prob = F.softmax(simStruc*self.gamma, dim=2)
        simClass = torch.sum(prob*simStruc, dim=2)

        simClass = simClass.view(N, H, W, self.K).transpose(2, 3).transpose(1, 2).contiguous()
        simClass = make_same_size(simClass, target)
        N, _, H, W = simClass.size()
        simClass = simClass.transpose(1, 2).transpose(2, 3).contiguous().view(N*H*W, self.K)
        target = target.view(N*H*W)

        marginM = torch.zeros(simClass.shape).cuda()
        marginM[torch.arange(0, marginM.shape[0]), target] = self.margin

        lossClassify = F.cross_entropy(self.la*(simClass-marginM), target, ignore_index=self.ignore_idx)
        if self.tau > 0 and self.K > 1:
            simCenter = centers.t().matmul(centers)
            reg = torch.sum(torch.sqrt(2.0+1e-5-2.*simCenter[self.weight]))/(self.cN*self.K*(self.K-1.))
            return lossClassify+self.tau*reg
        else:
            return lossClassify


class AreaLoss(nn.Module):
    def __init__(self, nclass, bigger=False):
        super(AreaLoss, self).__init__()
        self.bigger = bigger

    def forward(self, A, B):
        N, C, H, W = A.size()
        A = A.sigmoid().view(N, -1).float()
        B = B.view(N, -1).float()
        # B.area * range[0] < A.area < B.area * range[1]
        A_area = A.sum(1) / (H * W)
        B_area = B.sum(1) / (H * W)
        if self.bigger:
            area_loss = torch.clamp(A_area - B_area, 0)
        else:
            area_loss = torch.clamp(B_area - A_area , 0)

        area_loss = area_loss.mean()
        return area_loss


class CoverLoss(nn.Module):
    def __init__(self, nclass=1, radical=True, area_range=None):
        super(CoverLoss, self).__init__()
        self.radical = radical
        self.smooth = 1e-4
        self.area_range = area_range

    def forward(self, A, B):
        N, C, H, W = A.size()
        assert C == 1, "only support C == 1"
        A = A.sigmoid().view(N, -1).float()
        B = B.view(N, -1).float()

        if self.radical:
            loss = ((A * B).sum(1) + self.smooth) / (B.sum(1) + self.smooth)
        else:
            loss = ((A * B).sum(1) + self.smooth) / (A.sum(1) + self.smooth)
        cover_loss = 1 - loss.mean(0)

        area_loss = 0
        if self.area_range is not None:
            # B.area * range[0] < A.area < B.area * range[1]
            A_area = A.sum(1) / (H*W)
            B_area = B.sum(1) / (H*W)
            area_loss = torch.clamp(A_area - B_area*self.area_range[1], 0) \
                     + torch.clamp(B_area * self.area_range[0] - A_area, 0)
            area_loss = area_loss.mean()
            return cover_loss + area_loss
        else:
            return cover_loss


class F_beta(SingleLossWrapper):
    def __init__(self, nclass, beta=1):
        super(F_beta, self).__init__()
        self.beta_2 = beta**2

    def _calculate_loss(self, logits, target):
        N, C, H, W = logits.size()
        prob = logits.sigmoid().view(N, -1)
        target = target.view(N, -1).float()

        TP = (prob*target).sum(1)
        FN = ((1-prob)*target).sum(1)
        FP = ((1-target)*prob).sum(1)
        F_beta = (1+self.beta_2) * TP / ((1+self.beta_2)*TP + self.beta_2*FN + FP)
        return 1 - F_beta.mean(0)


class CenterLoss(nn.Module):
    def __init__(self, nclass):
        self.nclass = nclass

    def forward(self, logits, target):
        N, C, H, W = logits.size()
        probs = logits.softmax(1)
        preds = probs.argmax(1).expand(N, C, H, W)

        center_loss = 0
        for cls in range(self.nclass):
            cls_vectors = logits[preds == cls].view(N, -1)
            cls_prototype = cls_vectors.mean(1)
            cls_center_loss = torch.norm((cls_vectors - cls_prototype), p=2, dim=1).mean()
            center_loss += cls_center_loss
        return center_loss


@configs.LossFuncs.register('same_target')
def same_target_loss(loss_func_name, loss_num, *args, **kwargs):
    loss_func = configs.LossFuncs[loss_func_name](*args, **kwargs)
    return SameTargetLossWrapper(loss_num, loss_func)


@configs.LossFuncs.register('bce')
def bce_loss(**kwargs):
    return SingleLossWrapper(BCELoss(**kwargs))


@configs.LossFuncs.register('dice')
def dice_loss(**kwargs):
    return SingleLossWrapper(DiceLoss(**kwargs))


@configs.LossFuncs.register('sdice')
def simplified_dice(**kwargs):
    return SingleLossWrapper(SimplifiedDice(**kwargs))


@configs.LossFuncs.register('focal')
def focal_loss(**kwargs):
    return SingleLossWrapper(FocalLoss(**kwargs, gamma=1))


@configs.LossFuncs.register('dice_bound')
def dice_bound_loss(**kwargs):
    return SingleLossWrapper(BoundedDice(**kwargs))


@configs.LossFuncs.register('mse')
def mse_loss(**kwargs):
    return SingleLossWrapper(torch.nn.MSELoss())


@configs.LossFuncs.register('wce')
def weighted_crossentropy(**kwargs):
    return SingleLossWrapper(WeightedCrossEntropy(**kwargs))


@configs.LossFuncs.register('ce')
def weighted_crossentropy(**kwargs):
    return SingleLossWrapper(CrossEntropy(**kwargs))


@configs.LossFuncs.register('ohem_ce')
def ohem_crossentropy(**kwargs):
    return SingleLossWrapper(OhemCrossEntropy2d(**kwargs))


@configs.LossFuncs.register('none')
@configs.LossFuncs.register('empty')
def empty_loss(**kwargs):
    return SingleLossWrapper(EmptyLoss(**kwargs))