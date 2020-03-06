import functools

from torch import nn as nn

from network.components.evaluate_funcs import *
from utils.nn_utils import make_same_size, get_prediction, to_numpy

__all__ = ['LoggedMeasure', 'SegMeasure', 'MultiSegMeasure']


"""
Note that Measure cache can be any type.
"""


class LoggedMeasure(nn.Module):
    """ Like loggedLoss as a wrapper of loss function, is is also a
    wrapper for evaluation function to log evaluation.
    See **SegMeasure** for detail implementation.
    :param names : names for your outputs,
    :param main_acc_name : main measurement of evaluation ( of model performance). default : first name
    :param max_length : max history of evaluation
    """
    def __init__(self, nclass, names, main_acc_name=None, max_length=5000):
        super(LoggedMeasure, self).__init__()
        self.nclass = nclass
        self.names = names
        self.main_acc_axis = 0 if main_acc_name is None else self.names.index(main_acc_name)
        self.max_length = max_length
        self.pre_length = self.max_length
        self.caches = []
        self.mode = 'train'

    def clear_cache(self, mode='train'):
        self.caches.clear()
        self.mode = mode
        self._change_mode(mode)

    def _change_mode(self, mode):
        pass

    def append_eval(self, evaluation):
        """ Append evaluation to caches
        :param evaluation : the result of one evaluation
        """
        self.caches.append(evaluation)
        if self.training and len(self.caches) > self.max_length:
            self.caches.pop(0)

    @staticmethod
    def get_column(caches, idx):
        """ Get column from a 2-d(dim>=2) array
        :param caches:
        :param idx:
        :return:
        """
        column = []
        for cache in caches:
            column.append(cache[idx])
        return column

    def forward(self, logits, target):
        """ Calculate accuracy. Should be implemented by user.
        :param logits: output from network
        :param target: label from input
        """
        raise NotImplemented()

    def _get_acc(self):
        """ Should return the accuracy(average acc is recommended).
        """
        raise NotImplemented()

    def _get_last(self):
        """ Should return the last accuracy.
        """
        raise NotImplemented()

    def get_last(self):
        last_record = self._get_last()
        if not isinstance(last_record, (list, tuple)):
            last_record = (last_record,)
        assert len(last_record) == len(self.names), \
            "last record's length {} must match names length {}".format(len(last_record), len(self.names))
        return last_record

    def get_acc(self):
        acc = self._get_acc()
        if not isinstance(acc, (list, tuple)):
            acc = (acc,)
        assert len(acc) == len(self.names), \
            "accuracy's length {} must match names length {}".format(len(acc), len(self.names))
        return acc

    def get_acc_length(self):
        return len(self.names)

    def get_main_acc_idx(self):
        return self.main_acc_axis

    def get_acc_names(self, mode='train'):
        return self.names

    def get_main_acc(self):
        return self._get_acc()[self.main_acc_axis]

    def get_max_len(self):
        return self.max_length

    def set_max_len(self, new_len):
        self.pre_length = self.max_length
        self.max_length = new_len
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.max_length = self.pre_length


class EmptyMeasure(LoggedMeasure):
    def __init__(self, nclass):
        super(EmptyMeasure, self).__init__(nclass, names=['empty'])

    def forward(self, logits, target):
        return 0.,

    def _get_acc(self):
        return 0.,

    def _get_last(self):
        return 0.,


class SegMeasure(LoggedMeasure):
    """The channel of input logits should be the same as class number
    :param reduction  : the type of calculate accuracy of the result, {'immediate', 'delayed'},
                        'immediate' means average calculated acc,
                        'delayed'   means save the intersection and union, then calculate them
    :param main_acc_name : specify which accuracy to use as final evaluation, {'iou', 'dice'}
    :param multi_inputs_axis : if there is multiple outputs from the network,
                               use which output to compute result
    """

    def __init__(self, nclass, ignore_idx=-1, reduction='delayed', main_acc_name='iou',
                 multi_inputs_axis=-1, max_length=5000):
        names = ['iou', 'dice']
        super(SegMeasure, self).__init__(nclass, names, main_acc_name=main_acc_name, max_length=max_length)
        self.iou = functools.partial(mean_iou, ignore_idx=ignore_idx)
        self.dice = functools.partial(mean_dice, ignore_idx=ignore_idx)
        self.reduction = reduction
        self.multi_inputs_axis = multi_inputs_axis

    def forward(self, logits, target):
        if self.multi_inputs_axis != -1:
            # logits is a tuple object
            logits = logits[self.multi_inputs_axis]

        if isinstance(logits, list) or isinstance(logits, tuple):
            raise Exception("multi_inputs_axis must specified for multi outputs")

        logits = make_same_size(logits, target)
        if self.reduction == 'immediate':
            iou = self.iou(logits, target)
            dice = self.dice(logits, target)
            # calculate then average
            self.append_eval((iou.view(1), dice.view(1)))
        else:
            inter, union = get_inter_union(logits, target)
            # save then calculate
            self.append_eval((inter, union))

    def _get_acc(self):
        col_0 = self.get_column(self.caches, 0)
        col_1 = self.get_column(self.caches, 1)
        if self.reduction == 'immediate':
            ious, dices = col_0, col_1
            iou = torch.cat(ious, 0)
            dice= torch.cat(dices, 0)
        else:
            # size : C
            inter, union = col_0, col_1
            class_inter = torch.cat(inter, 0)
            class_union = torch.cat(union, 0)
            iou = IOU(class_inter, class_union)
            dice = DICE(class_inter, class_union)
        return iou.mean().item(), dice.mean().item()

    def _get_last(self):
        last_cache = self.caches[-1]
        if self.reduction == 'immediate':
            iou, dice = last_cache
        else:
            inter, union = last_cache
            iou = IOU(inter, union)
            dice = DICE(inter, union)
        return iou.mean().item(), dice.mean().item()


class SimpleMeasure(LoggedMeasure):
    def __init__(self, nclass, name, func, max_length=500):
        super(SimpleMeasure, self).__init__(nclass, [name], max_length=max_length)
        self.func = func

    def forward(self, logits, target):
        acc = self.func(logits, target)
        self.append_eval(acc)

    def _get_acc(self):
        return torch.stack(self.caches, 0).mean(),

    def _get_last(self):
        return self.caches[-1],


class DiceMeasure(LoggedMeasure):

    def __init__(self, nclass, name='dice', reduction='delayed', max_length=5000):
        name = name.lower()
        if name == 'iou':
            names = ['iou']
            self.func = mean_iou
            self.delay_func = IOU
        elif name == 'dice':
            names = ['dice']
            self.func = mean_dice
            self.delay_func = DICE
        else:
            raise Exception("Error name : {}".format(name))

        super(DiceMeasure, self).__init__(nclass, names, max_length=max_length)
        self.reduction = reduction

    def forward(self, logits, target):
        """
        :type  logits: torch.Tensor : size is [N, C, D, H, W]
        :type target: torch.Tensor : size is [N, 1, D, H, W]
        """
        logits = make_same_size(logits, target)
        if self.reduction == 'immediate':
            acc = self.func(logits, target)
            # calculate then average
            self.append_eval((acc.view(1),))
        else:
            inter, union = get_inter_union(logits, target)
            # save then calculate
            self.append_eval((inter, union))

    def _get_acc(self):
        if self.reduction == 'immediate':
            acc_list = self.get_column(self.caches, 0)
            acc = torch.cat(acc_list, 0)
        else:
            col_0 = self.get_column(self.caches, 0)
            col_1 = self.get_column(self.caches, 1)
            # size : C
            inter, union = col_0, col_1
            class_inter = torch.cat(inter, 0)
            class_union = torch.cat(union, 0)
            acc = self.delay_func(class_inter, class_union)
        return acc.mean().item(),

    def _get_last(self):
        last_cache = self.caches[-1]
        if self.reduction == 'immediate':
            acc = last_cache
        else:
            inter, union = last_cache
            acc = self.delay_func(inter, union)
        return acc.mean().item(),


class MultiSegMeasure(LoggedMeasure):
    """
    :type list[LoggedMeasure] measures : a list[LoggedMeasure]
    """
    def __init__(self, nclass, measures, main_acc_name=None, max_length=5000):
        self.measures = measures
        self.names = []
        for measure in self.measures:
            self.names += measure.names
        super(MultiSegMeasure, self).__init__(nclass, self.names, main_acc_name=main_acc_name, max_length=max_length)

    def forward(self, logits, target):
        assert len(logits) == len(target) == len(self.measures)

        # target[0][target[0]==2] = 1
        for measure, l, t in zip(self.measures, logits, target):
            measure(l, t)

    def _get_acc(self):
        acc_list = []
        for measure in self.measures:
            for acc in measure._get_acc():
                acc_list.append(acc)
        return acc_list

    def _get_last(self):
        acc_list = []
        for measure in self.measures:
            for acc in measure._get_last():
                acc_list.append(acc)
        return acc_list

    def clear_cache(self, mode='train'):
        for measure in self.measures:
            measure.clear_cache(mode)


class Class3SegMeasure(SegMeasure):
    def __init__(self, nclass, reduction='delayed', main_acc_name='iou', max_length=5000):
        super(Class3SegMeasure, self).__init__(nclass, main_acc_name=main_acc_name, reduction=reduction, max_length=max_length)

    def forward(self, logits, target):
        # (background + kidney) is all background , 2 is tumor, 1 is kidney
        if target.max() < 2:
            new_logits = logits[:, [0, 2], ...]
            # add kidney prediction to background
            new_logits[:, 0, ...] += logits[:, 1, ...]
        else:
            new_logits = logits
        super(Class3SegMeasure, self).forward(new_logits, target)


class MeterDicts(dict):
    def __init__(self, averaged=[]):
        if not isinstance(averaged, (list, tuple)):
            self.averaged = [averaged]
        else:
            self.averaged = averaged

    def update_meters(self, dicts, skips=[]):
        if len(list(self.keys())) == 0:
            for k, v in dicts.items():
                self[k] = AverageMeter().update(v)
        else:
            for k, v in dicts.items():
                if k not in skips:
                    if k in self.averaged:
                        self[k].reset()
                        self[k].update(v)
                    else:
                        self[k].update(v)
        return self


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self


class ClassificationMeasure(LoggedMeasure):
    def __init__(self):
        super(ClassificationMeasure, self).__init__(nclass=1, names=['acc'])
        self.acc_meter = AverageMeter()

    def forward(self, logits, target):
        assert len(logits.size()) - len(target.size()) == 1, "logits.size : {}, target.size : {}".format(logits.size(), target.size())
        acc = (logits.argmax(1) == target).sum().float() / target.size(0)
        self.acc_meter.update(acc.item())

    def clear_cache(self, mode='train'):
        self.acc_meter.reset()

    def get_acc(self):
        return self.acc_meter.avg

    def get_last(self):
        return self.acc_meter.val


class NaturalImageMeasure(LoggedMeasure):
    def __init__(self, nclass, distributed=False, max_length=500, cpu=False):
        super(NaturalImageMeasure, self).__init__(nclass, names=['mIoU', 'acc', 'FWIoU'],
                                                  main_acc_name='mIoU', max_length=max_length)
        self.inter_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.freq_meter = AverageMeter()
        self.total_meter = AverageMeter()
        self.dist = distributed
        self.cpu = cpu

    def clear_cache(self, mode='train'):
        self.inter_meter.reset()
        self.union_meter.reset()
        self.freq_meter.reset()
        self.total_meter.reset()

    def get_something(self, matrix):
        if not self.cpu:
            inter = matrix.diagonal().contiguous()
            union = (matrix.sum(0) + matrix.sum(1) - inter).contiguous()
            total = matrix.sum().contiguous()
            freq = (matrix.sum(1) / total).contiguous()
        else:
            inter = matrix.diagonal()
            union = (matrix.sum(0) + matrix.sum(1) - inter)
            total = matrix.sum()
            freq = (matrix.sum(1) / total)
        return inter, union, total, freq

    def forward(self, logits, target):
        logits = make_same_size(logits, target)
        prediction = get_prediction(logits, cpu=self.cpu)
        if len(target.shape) != len(prediction.shape):
            target = target[:, 0]

        if self.cpu:
            current_matrix = get_confusion_matrix(prediction, target, self.nclass)
        else:
            current_matrix = get_confusion_matrix_gpu(prediction, target, self.nclass)
        inter, union, total, freq = self.get_something(current_matrix)
        inter, union, total, freq = to_numpy([inter, union, total, freq])
        self.inter_meter.update(inter), self.union_meter.update(union)
        self.total_meter.update(total), self.freq_meter.update(freq)

    def get_res(self, last=False):
        iou, acc, freq = self.get_class_acc(last)
        mIoU = np.nanmean(iou)
        fwIoU = np.nansum(freq * iou)
        return mIoU, acc, fwIoU

    def get_class_acc(self, last=False):
        if not last:
            inter = self.inter_meter.sum
            union = self.union_meter.sum
            total = self.total_meter.sum
            freq  = self.freq_meter.sum
        else:
            inter = self.inter_meter.val
            union = self.union_meter.val
            total = self.total_meter.val
            freq  = self.freq_meter.val

        iou = inter / union
        acc = inter.sum() / total
        return iou, acc, freq

    def get_acc(self):
        return self.get_res(last=False)

    def get_last(self):
        return self.get_res(last=True)


class AxisMeasureWrapper(LoggedMeasure):
    """ A measurement wrapper to calculate multi logits and target.
    e.g. logits = [l1, l2, l3]
         target = t1

         axis = 0
         => calculate (l1, t1)

    e.g. logits = [l1, l2, l3]
         target = [t1, t2, t3]

         axis = 1
         => calculate (l2, t2)
    """
    def __init__(self, measure:LoggedMeasure, axis=0):
        super(AxisMeasureWrapper, self).__init__(nclass=measure.nclass, names=measure.names,
                                                 main_acc_name=measure.names[measure.main_acc_axis],
                                                 max_length=measure.max_length)
        self.measure = measure
        self.axis = axis

    def forward(self, logits, target):
        if isinstance(target, (tuple, list)):
            target = target[self.axis]
        if isinstance(logits, (tuple, list)):
            logits = logits[self.axis]
        return self.measure.forward(logits, target)

    def get_acc(self):
        return self.measure.get_acc()

    def get_acc_names(self, mode='train'):
        return self.measure.get_acc_names(mode)

    def get_last(self):
        return self.measure.get_last()

    def clear_cache(self, mode='train'):
        self.measure.clear_cache(mode)



if __name__ == '__main__':
    pred = np.array( [[1, 1, 0],
                      [1, 1, 0],
                      [0, 0, 0]])

    ground = np.array([[0, 0, 0],
                       [0, 1, 1],
                       [0, 1, 1]])
    matrix = get_confusion_matrix(pred, ground, 2)
    print(matrix)
