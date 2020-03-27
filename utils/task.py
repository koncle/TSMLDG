import atexit
import functools
import itertools
import os
import time
import traceback
import warnings

from yacs.config import CfgNode

"""
# install package 
pip install nvidia-ml-py3

[Task] is the class to create cfg and param for task.

[Schedulable] is an interface to be scheduled by the method...
[Job] is a class to manage the schedulable objects

[CfgJob] is the class to get cfg and param for lots of tasks, and schedule these cfgs to specified gpus.
[FunctionJob] is the class to schedule any functions. 
"""
import nvidia_smi


# pip install nvidia-ml-py3
class NvidiaSmi():
    total_devices = 0
    init = False

    def __enter__(self):
        if not NvidiaSmi.init:
            nvidia_smi.nvmlInit()
            NvidiaSmi.total_devices = nvidia_smi.nvmlDeviceGetCount()
            NvidiaSmi.init = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        nvidia_smi.nvmlShutdown()

    @staticmethod
    def get_gpu_memory(device_idx):
        assert device_idx < NvidiaSmi.total_devices, "device index should {} less than total devices {}"\
            .format(device_idx, NvidiaSmi.total_devices)
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_idx)
        res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        M = 1024**2
        return res.free / M, res.total / M, res.used / M

    @staticmethod
    def get_gpu_utility(device_idx):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_idx)
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        return res.memory, res.gpu

    @staticmethod
    def get_free_gpus(threshold=8000, gpu_ids=[]):
        free_gpu_idx = []
        free_gpus = 0
        for idx in range(NvidiaSmi.total_devices):
            if len(gpu_ids) == 0 or idx in gpu_ids:
                memory = NvidiaSmi.get_gpu_memory(idx)[0]
                if memory > threshold:
                    free_gpu_idx.append(idx)
                    free_gpus += 1
        return free_gpus, free_gpu_idx

    @staticmethod
    def print_device_info():
        for i in range(NvidiaSmi.total_devices):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            print("Device {}, : {}".format(i, nvidia_smi.nvmlDeviceGetName(handle)))


def change_value(cfg: CfgNode, key: str, value):
    """ Change cfg value with key and value """
    parent_key, child_key = key.split('.')
    cfg[parent_key][child_key] = value
    return cfg


def get_value(cfg: CfgNode, key: str):
    parent_key, child_key = key.split('.')
    return cfg[parent_key][child_key]


class Cfg(object):
    """
    An object to provide cfg and target values,
    so that user can create a task easier
    """

    def __init__(self, cfg_name):
        self.cfg_name = cfg_name
        self.target_values = {}

    def add_values(self, key: str, value, is_param=False, foreach=False):
        """ write new (key, value) pair to config.
        :param key: key of cfg, form of 'trainer.net'
        :param value: the value of the key
        :param is_param: if true this value will be considered as a combination of other values,
                         for example, if we have two key-value pair (k1, [1, 2]), (k2, [3, 4]),
                         if they are all params, then, four cfg will be generated in which the value is
                         [[1, 3], [1, 4], [2, 3], [2, 4]], then the key-value will be
                         [ [(k1: 1), (k2:3)], [(k1:1), (k2:4)], ...]
        :param foreach: only non-param have effect of this value, if true, the param should be a list,
                        and it will be assigned to every generated cfg.
        :return:
        """
        key = key.strip()
        if is_param and not isinstance(value, (list, tuple)):
            warnings.warn('param value should be list, put it into list automatically')
            value = [value]
        elif not is_param and not foreach and isinstance(value, (list, tuple)):
            warnings.warn('non-param value should not be list, do you forget to set foreach to True ?')
        self.target_values[key] = {'values': value, 'is_param': is_param, 'foreach': foreach}

    def copy(self):
        task = Cfg(self.cfg_name)
        task.target_values = self.target_values.copy()
        return task

    def create_cfgs(self):
        """ Create cfg according to task"""
        origin_cfg = load_cfg(self.cfg_name)

        param_keys = []
        fixed_keys = []
        param_values = []
        fixed_values = []
        foreachs = []
        for k, v in self.target_values.items():
            if v['is_param']:
                param_values.append(v['values'])
                param_keys.append(k)
            else:
                fixed_values.append(v['values'])
                foreachs.append(v['foreach'])
                fixed_keys.append(k)
        # rearrange keys.
        keys = param_keys + fixed_keys
        # get combinations of all params
        param_combinations = itertools.product(*param_values)
        final_combinations = []
        for i, params in enumerate(param_combinations):
            params = list(params)
            for j, value in enumerate(fixed_values):
                if foreachs[j]:
                    params.append(value[i])
                else:
                    params.append(value)
            final_combinations.append(params)

        # overwrite params in provided cfg
        cfgs = []
        param_strings = []
        for params in final_combinations:
            param_string = ''
            new_cfg = origin_cfg.clone()
            for k, v in zip(keys, params):
                new_cfg = change_value(new_cfg, k, v)
                if 'stage' not in k and 'gpu' not in k and 'test_epoch' not in k and 'root' not in k:
                    param_string += '_' + str(v)
            if 'trainer.output_path' not in keys:
                new_cfg = change_value(new_cfg, 'trainer.output_path', new_cfg['trainer']['output_path'] + param_string)
            cfgs.append(new_cfg)
            param_strings.append(param_string)

        # if no cfg generated, use default cfg
        if not cfgs:
            cfgs.append(origin_cfg)
            param_strings.append('Default')
        return cfgs, param_strings


class Schedulable(object):
    """ If the function want to be scheduled,
        it should implement this method
    """

    def __init__(self, name, gpus):
        self.name = name
        self.gpus = gpus

        self.status = None
        self.gpu_list = None
        self.pid = None

    def set_gpu_idx(self, gpu_ids):
        self.gpu_list = gpu_ids

    def set_pid(self, pid, status):
        self.pid = pid
        self.status = status

    def get_pid(self):
        return self.pid

    def run(self):
        """
        Status[pid] = True or False. When the task finished.
        :return:
        """
        raise NotImplementedError()

    def __repr__(self):
        return self.name

    def get_gpus(self):
        return self.gpus


# shared variable for multi-processes
from multiprocessing import Manager

manager = Manager()
states = manager.dict()


def schedule_tasks(tasks: [Schedulable], sync=False, process_interval=1, minimum_memory=8000, gpu_ids=[]):
    from multiprocessing import Process
    import time
    import random

    states.clear()
    task_ids = {task: i for i, task in enumerate(tasks)}
    running_tasks = []
    print('Waiting for GPU.')
    with NvidiaSmi() as nv:
        processes = []
        while len(tasks) > 0:

            # If not synchronize process or there is no process running.
            # Add more process.
            if not sync or len(processes) == 0:
                # random sleep to prevent Process crash to OOM in multi processes
                if len(processes) != 0:
                    time.sleep(random.randint(0, process_interval))

                # get free gpu index
                free_num, free_gpu_idx_list = nv.get_free_gpus(threshold=minimum_memory, gpu_ids=gpu_ids)
                free_gpu_idx_list = [str(idx) for idx in free_gpu_idx_list]

                # iterate cfgs, find suitable cfg to execute
                for task in tasks:
                    gpu_num = len(task.get_gpus())

                    # allocate gpu for this task
                    if free_num >= gpu_num:
                        task.set_gpu_idx(free_gpu_idx_list[:gpu_num])
                        gpu_str = ','.join(free_gpu_idx_list[:gpu_num])

                        # do
                        task.set_pid(task_ids[task], states)
                        p = Process(target=task.run)
                        p.start()
                        processes.append(p)
                        print('Allocated task {} to gpu : {}'.format(str(task), gpu_str))

                        running_tasks.append(task)
                        tasks.remove(task)
                        break

                # sleep after scheduled a task
                time.sleep(process_interval)

                # remove completed tasks
                for task in running_tasks:
                    task_pid = task.get_pid()
                    if task_pid in states and states[task_pid] is True:
                        running_tasks.remove(task)

        # wait for all processes
        for p in processes:
            p.join()

        # all tasks have been scheduled
        # check running task again to make sure all tasks have been executed correctly

        for task in running_tasks:
            task_pid = task.get_pid()
            if task_pid not in states or states[task_pid] is False:
                tasks.append(task)

        if len(tasks) > 0:
            print('Schedule failed tasks again : [{}]'.format(str([str(t) for t in tasks])))
            schedule_tasks(tasks, sync, process_interval, minimum_memory)


class SchedulableTask(Schedulable):
    def __init__(self, cfg, param, gpus):
        super(SchedulableTask, self).__init__(param, gpus)
        self.cfg = cfg
        self.param = param
        self.gpu_list = gpus

    def run(self):
        """ Train networks with cfgs generated from task"""
        gpu_value = ','.join(self.gpu_list)
        change_value(self.cfg, 'trainer.gpus', gpu_value)

        done = True
        try:
            import os
            trainer = Trainer(self.cfg)
            trainer.train()
        except RuntimeError as e:
            traceback.print_exc()
            if 'out of memory' in str(e):
                done = False
                print('task will be redone if it is called with schedule_task() ')
            else:
                raise e
        finally:
            atexit._run_exitfuncs()
            states[self.pid] = done


class Job(object):
    def run(self, sync=False, process_interval=10, minimum_memory=8000):
        raise NotImplementedError


class CfgJob(Job):
    def __init__(self, cfgs: [Cfg]):
        super(CfgJob, self).__init__()
        cfg_list, param_list = self.create_cfg_and_params_from_tasks(cfgs)
        self.schedulable_tasks = self.create_scheduable_tasks(cfg_list, param_list)

    def create_cfg_and_params_from_tasks(self, cfgs):
        if not isinstance(cfgs, list):
            cfgs = [cfgs]
        cfg_list, param_list = [], []
        for task in cfgs:
            cfgs, params = task.create_cfgs()
            cfg_list.extend(cfgs)
            param_list.extend(params)
        return cfg_list, param_list

    def create_scheduable_tasks(self, cfg_list, param_list):
        scheduable_tasks = []
        for cfg, param in zip(cfg_list, param_list):
            gpus = get_value(cfg, 'trainer.gpus').split(',')
            t = SchedulableTask(cfg, param, gpus)
            scheduable_tasks.append(t)
        return scheduable_tasks

    def run(self):
        for task in self.schedulable_tasks:
            task.run()

    def parallel_run(self, sync=False, process_interval=10, minimum_memory=8000, gpu_ids=[]):
        schedule_tasks(self.schedulable_tasks, sync, process_interval, minimum_memory, gpu_ids=gpu_ids)


class SchedulableFunc(Schedulable):
    def __init__(self, func, gpus):
        super(SchedulableFunc, self).__init__(str(func), gpus)
        self.func = func

    def run(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(self.gpu_list)
        self.func()
        self.status[self.pid] = True


class FunctionJob(Job):
    def __init__(self, funcs, gpus):
        if not isinstance(funcs, (tuple, list)):
            funcs = [funcs]
        if not isinstance(gpus, (tuple, list)):
            gpus = [gpus]
        assert len(gpus) == len(funcs)

        self.funcs = funcs
        self.gpus = gpus
        self.tasks = self.create_schedulable_tasks()

    def create_schedulable_tasks(self):
        tasks = []
        for f, g in zip(self.funcs, self.gpus):
            t = SchedulableFunc(f, g)
            tasks.append(t)
        return tasks

    def run(self, sync=False, process_interval=10, minimum_memory=8000):
        schedule_tasks(self.tasks, sync, process_interval, minimum_memory)
