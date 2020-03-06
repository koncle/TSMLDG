import torch
from torch.optim import SGD
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from engine import configs

import network.components.schedulers
import network.components.customized_loss
import network.components.customized_evaluate as Eval

import network.nets
import dataset
import engine.logger
import numpy as np
import names_to_be_imported

"""
DO! NOT! DELETE! the imports above. 
"""

__all__ = ['get_network', 'get_solver', 'get_dataloader',
           'get_loss_func', 'get_eval_func', 'get_logger_manager', 'get_scheduler']


def get_function(registry, name):
    if name in registry:
        print('[{}] is defined in {}'.format(name, registry.get_src_file(name)))
        return registry[name]
    else:
        raise Exception("{} does not support [{}], valid keys : {}".format(registry, name, list(registry.keys())))


def get_network(cfg):
    name = cfg.model.net
    in_ch = cfg.model.in_ch
    nclass = cfg.datasets.nclass
    args = eval(cfg.model.kwargs)
    args['in_ch'] = in_ch
    args['nclass'] = nclass
    func = get_function(configs.Networks, name)
    return func(**args)


def get_solver(net, loss, cfg):
    opt_name = cfg.solver.optimizer.lower()
    lr = cfg.solver.base_lr
    wd = cfg.solver.weight_decay

    if hasattr(net, 'get_parameters'):
        params = net.get_parameters(lr)
    else:
        params = [{'params': net.parameters()}]

    if hasattr(loss, 'get_parameters'):
        params += loss.get_parameters(lr)
    else:
        params += [{'params': loss.parameters()}]

    if opt_name == 'adam':
        return Adam(lr=lr, params=params, weight_decay=wd)
    elif opt_name == 'sgd':
        momentum = cfg.solver.momentum
        return SGD(params=params, lr=lr, weight_decay=wd, momentum=momentum)
    else:
        raise Exception("Not support opt : {}".format(opt_name))


def get_scheduler(opt, dataloader, cfg):
    scheduler_name = cfg.scheduler.name.lower()
    epoch = cfg.trainer.epoch
    iteration = len(dataloader)
    warmup_epochs = cfg.scheduler.warmup_epochs
    iteration_decay = cfg.scheduler.iteration_decay

    kwargs = eval(cfg.scheduler.kwargs)
    if not isinstance(kwargs, dict):
        raise Exception("scheduler args should be string of dict, e.g. '{k1:v1}'")

    args = {
        'optimizer': opt,
        'total_epoch': epoch,
        'iteration_per_epoch': iteration,
        'warmup_epochs': warmup_epochs,
        'iteration_decay': iteration_decay
    }
    func = get_function(configs.Schedulers, scheduler_name)
    return func(**args, **kwargs)


def get_loss_func(cfg) -> network.components.customized_loss.LoggedLoss:
    loss_func_name = cfg.evaluate.loss_func
    nclass = cfg.datasets.nclass

    loss_func_kwargs = eval(cfg.evaluate.loss_kwargs)
    loss_func_kwargs['nclass'] = nclass
    if not isinstance(loss_func_kwargs, dict):
        raise Exception("Loss func args should be string of dict, e.g. '{k1:v1}'")

    func = get_function(configs.LossFuncs, loss_func_name)
    loss = func(**loss_func_kwargs)

    assert isinstance(loss, network.components.customized_loss.LoggedLoss), \
        "Either SingleLossWrapper() or MultiLossWrapper() should be used " \
        "or inherit LoggedLoss() when registering a loss function"

    return loss


def get_eval_func(cfg) -> Eval.SegMeasure:
    eval_func_name = cfg.evaluate.eval_func
    eval_func_kwargs = eval(cfg.evaluate.eval_kwargs)
    distributed = cfg.cuda.distributed
    eval_func_kwargs.update({'nclass': cfg.datasets.nclass, 'distributed': distributed})

    if not isinstance(eval_func_kwargs, dict):
        raise Exception("Eval func args should be string of dict, e.g. '{k1:v1}'")

    func = get_function(configs.EvalFuncs, eval_func_name)
    return func(**eval_func_kwargs)


def get_transforms(transforms):
    is_3d = transforms.is_3d
    random_crop = transforms.random_crop
    random_scale = transforms.random_scale
    random_flip = transforms.random_flip
    return None


def get_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test', 'full', 'pred']
    dataset = None
    output_path = cfg.trainer.output_path
    force_cache = cfg.datasets.force_cache
    if mode != 'train':
        force_cache = False

    if mode in ['train', 'val', 'test']:
        root = cfg.datasets.root
        name = cfg.datasets.name
        origin_kwargs = {
            'mode': mode,
            'output_path': output_path,
            'root': root,
            'force_cache' : force_cache
        }
        new_kwargs = eval(cfg.datasets.kwargs)
        origin_kwargs.update(new_kwargs)
        func = get_function(configs.Datasets, name)
        dataset = func(**origin_kwargs)

    elif mode == 'full':
        # full image loader should load full image to train instead of patches from test set.
        kwargs = {'mode': 'test'}
        kwargs.update(eval(cfg.datasets.full_kwargs))
        root = cfg.datasets.full_root
        name = cfg.datasets.full_dataset
        if root.strip() == '' or name.strip() == '':
            dataset = None
        else:
            func = get_function(configs.Datasets, name)
            dataset = func(output_path=output_path, root=root, force_cache=force_cache, **kwargs)

    elif mode == 'pred':
        kwargs = {'mode': 'test'}
        kwargs.update(eval(cfg.datasets.pred_kwargs))
        root = cfg.datasets.pred_root
        name = cfg.datasets.pred_dataset
        if root.strip() == '' or name.strip() == '':
            dataset = None
        else:
            func = get_function(configs.Datasets, name)
            dataset = func(output_path=output_path, root=root, force_cache=force_cache, **kwargs)

    return dataset


def worker_init_fn(_):
    return np.random.seed(int(torch.initial_seed()) % (2 ** 32 - 1))


def get_dataloader(cfg, modes, distributed=False, rank=None):
    shuffle = cfg.dataloader.shuffle
    drop_last = cfg.dataloader.drop_last
    num_workers = cfg.dataloader.num_workers
    test_batch_size = cfg.dataloader.test_batch_size
    train_batch_size = cfg.dataloader.train_batch_size

    if distributed:
        rank_num = cfg.cuda.dist_ranks
        train_batch_size = train_batch_size // rank_num
        test_batch_size = test_batch_size // rank_num
        shuffle = False

    loaders = []
    for i, mode in enumerate(modes):
        dataset = get_dataset(cfg, mode)

        if dataset is not None and distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank)
        else:
            sampler = None

        if mode == 'train':
            loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=True, drop_last=drop_last, sampler=sampler,
                                worker_init_fn=worker_init_fn)
        elif mode in ['val', 'test']:
            loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=drop_last, sampler=sampler,
                                worker_init_fn=worker_init_fn)
        elif mode in ['full', 'pred']:
            if dataset is not None:
                loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                    num_workers=num_workers, pin_memory=True, drop_last=drop_last, sampler=sampler,
                                    worker_init_fn=worker_init_fn)
            else:
                loader = None

        loaders.append(loader)
    return loaders


def get_logger_manager(cfg, component_state, log_state, rank=0):
    logger_names = cfg.trainer.loggers

    args = {
        'cfg': cfg,
        'component_state': component_state,
        'log_state': log_state
    }
    if 'file' not in logger_names:
        logger_names.append('file')

    kwargs_list = cfg.trainer.loggers_kwargs
    if not kwargs_list:
        kwargs_list = ["{}"] * len(logger_names)
    else:
        assert len(kwargs_list) == len(logger_names), \
            'the length of logger kwargs :{} should be the same as the length of loggers :{}' \
                .format(len(kwargs_list), len(logger_names))

    loggers = []
    for name, kwargs in zip(logger_names, kwargs_list):
        args.update(eval(kwargs))
        logger = get_function(configs.Loggers, name)(**args)
        loggers.append(logger)
    logger_manager = engine.logger.LoggerManager(loggers, rank)
    return logger_manager
