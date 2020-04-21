from tensorboardX import SummaryWriter
from torch import nn
from torch.optim import SGD
from tqdm import tqdm

from dataset.dg_dataset import *
from network.components.customized_evaluate import NaturalImageMeasure, MeterDicts
from network.components.schedulers import PolyLR
from resnet import Net
from utils.nn_utils import *
from utils.nn_utils import get_updated_network, get_logger, get_img_target
from utils.visualize import show_graphs

# 123456
seed = 123456
torch.manual_seed(seed)
np.random.seed(seed)


class MetaFrameWork(object):
    def __init__(self, name='normal_all', train_num=1, source='GSIM',
                 target='C', network=Net, resume=True, dataset=DGMetaDataSets,
                 inner_lr=1e-3, outer_lr=5e-3, train_size=8, test_size=16, no_source_test=True, bn='torch'):
        super(MetaFrameWork, self).__init__()
        self.no_source_test = no_source_test
        self.train_num = train_num
        self.exp_name = name
        self.resume = resume

        self.inner_update_lr = inner_lr
        self.outer_update_lr = outer_lr
        self.network = network
        self.dataset = dataset
        self.train_size = train_size
        self.test_size = test_size
        self.source = source
        self.target = target
        self.bn = bn

        self.epoch = 1
        self.best_target_acc = 0
        self.best_target_acc_source = 0
        self.best_target_epoch = 1

        self.best_source_acc = 0
        self.best_source_acc_target = 0
        self.best_source_epoch = 0

        self.total_epoch = 120
        self.save_interval = 1
        self.save_path = Path(self.exp_name)
        self.init()

    def init(self):
        kwargs = {'bn': self.bn, 'output_stride': 8}
        self.backbone = nn.DataParallel(self.network(**kwargs)).cuda()
        kwargs.update({'pretrained': False})
        self.updated_net = nn.DataParallel(self.network(**kwargs)).cuda()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        self.nim = NaturalImageMeasure(nclass=19)

        batch_size = self.train_size
        workers = len(self.source) * 4

        dataloader = functools.partial(DataLoader, num_workers=workers, pin_memory=True, batch_size=batch_size, shuffle=True)
        self.train_loader = dataloader(self.dataset(mode='train', domains=self.source, force_cache=True))

        dataloader = functools.partial(DataLoader, num_workers=workers, pin_memory=True, batch_size=self.test_size, shuffle=False)
        self.source_val_loader = dataloader(self.dataset(mode='val', domains=self.source, force_cache=True))

        target_dataset, folder = get_dataset(self.target)
        self.target_loader = dataloader(target_dataset(root=ROOT + folder, mode='val'))
        self.target_test_loader = dataloader(target_dataset(root=ROOT + 'cityscapes', mode='test'))

        self.opt_old = SGD(self.backbone.parameters(), lr=self.outer_update_lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler_old = PolyLR(self.opt_old, self.total_epoch, len(self.train_loader), 0, True, power=0.9)

        self.logger = get_logger('train', self.exp_name)
        self.log('exp_name : {}, train_num = {}, source domains = {}, target_domain = {}, lr : inner = {}, outer = {},'
                 'dataset : {}, net : {}, bn : {}\n'.
                 format(self.exp_name, self.train_num, self.source, self.target, self.inner_update_lr, self.outer_update_lr, self.dataset,
                        self.network, self.bn))
        self.log(self.exp_name + '\n')
        self.train_timer, self.test_timer = Timer(), Timer()

    def train(self, epoch, it, inputs):
        # imgs : batch x domains x C x H x W
        # targets : batch x domains x 1 x H x W
        imgs, targets = inputs
        B, D, C, H, W = imgs.size()
        meta_train_imgs = imgs.view(-1, C, H, W)
        meta_train_targets = targets.view(-1, 1, H, W)

        tr_logits = self.backbone(meta_train_imgs)[0]
        tr_logits = make_same_size(tr_logits, meta_train_targets)
        ds_loss = self.ce(tr_logits, meta_train_targets[:, 0])
        with torch.no_grad():
            self.nim(tr_logits, meta_train_targets)

        self.opt_old.zero_grad()
        ds_loss.backward()
        self.opt_old.step()
        self.scheduler_old.step(epoch, it)
        losses = {
            'dg': 0,
            'ds': ds_loss.item()
        }
        acc = {
            'iou': self.nim.get_res()[0]
        }
        return losses, acc, self.scheduler_old.get_lr(epoch, it)[0]

    def meta_train(self, epoch, it, inputs):
        # imgs : batch x domains x C x H x W
        # targets : batch x domains x 1 x H x W

        imgs, targets = inputs
        B, D, C, H, W = imgs.size()
        split_idx = np.random.permutation(D)
        i = np.random.randint(1, D)
        train_idx = split_idx[:i]
        test_idx = split_idx[i:]
        # train_idx = split_idx[:D // 2]
        # test_idx = split_idx[D // 2:]

        # self.print(split_idx, B, D, C, H, W)'
        meta_train_imgs = imgs[:, train_idx].reshape(-1, C, H, W)
        meta_train_targets = targets[:, train_idx].reshape(-1, 1, H, W)
        meta_test_imgs = imgs[:, test_idx].reshape(-1, C, H, W)
        meta_test_targets = targets[:, test_idx].reshape(-1, 1, H, W)

        # Meta-Train
        tr_logits = self.backbone(meta_train_imgs)[0]
        tr_logits = make_same_size(tr_logits, meta_train_targets)
        ds_loss = self.ce(tr_logits, meta_train_targets[:, 0])

        # Update new network
        self.opt_old.zero_grad()
        ds_loss.backward(retain_graph=True)
        updated_net = get_updated_network(self.backbone, self.updated_net, self.inner_update_lr).train().cuda()

        # Meta-Test
        te_logits = updated_net(meta_test_imgs)[0]
        # te_logits = test_res[0]
        te_logits = make_same_size(te_logits, meta_test_targets)
        dg_loss = self.ce(te_logits, meta_test_targets[:, 0])
        with torch.no_grad():
            self.nim(te_logits, meta_test_targets)

        # Update old network
        dg_loss.backward()
        self.opt_old.step()
        self.scheduler_old.step(epoch, it)
        losses = {
            'dg': dg_loss.item(),
            'ds': ds_loss.item()
        }
        acc = {
            'iou': self.nim.get_res()[0],
        }
        return losses, acc, self.scheduler_old.get_lr(epoch, it)[0]

    def do_train(self):
        if self.resume:
            self.load()

        self.writer = SummaryWriter(str(self.save_path / 'tensorboard'), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S'))
        self.log('Start epoch : {}\n'.format(self.epoch))

        for epoch in range(self.epoch, self.total_epoch + 1):
            loss_meters, acc_meters = MeterDicts(), MeterDicts(averaged=['iou'])
            self.nim.clear_cache()
            self.backbone.train()
            self.epoch = epoch
            with self.train_timer:
                for it, (paths, imgs, target) in enumerate(self.train_loader):
                    meta = (it + 1) % self.train_num == 0
                    if meta:
                        losses, acc, lr = self.meta_train(epoch - 1, it, to_cuda([imgs, target]))
                    else:
                        losses, acc, lr = self.train(epoch - 1, it, to_cuda([imgs, target]))

                    loss_meters.update_meters(losses, skips=['dg'] if not meta else [])
                    acc_meters.update_meters(acc)

                    self.print(self.get_string(epoch, it, loss_meters, acc_meters, lr, meta), end='')
                    self.tfb_log(epoch, it, loss_meters, acc_meters)
            self.print(self.train_timer.get_formatted_duration())
            self.log(self.get_string(epoch, it, loss_meters, acc_meters, lr, meta) + '\n')

            self.save('ckpt')
            if epoch % self.save_interval == 0:
                with self.test_timer:
                    city_acc = self.val(self.target_loader)
                    self.save_best(city_acc, epoch)

            total_duration = self.train_timer.duration + self.test_timer.duration
            self.print('Time Left : ' + self.train_timer.get_formatted_duration(total_duration * (self.total_epoch - epoch)) + '\n')

        self.log('Best city acc : \n  city : {}, origin : {}, epoch : {}\n'.format(
            self.best_target_acc, self.best_target_acc_source, self.best_target_epoch))
        self.log('Best origin acc : \n  city : {}, origin : {}, epoch : {}\n'.format(
            self.best_source_acc_target, self.best_source_acc, self.best_source_epoch))

    def save_best(self, city_acc, epoch):
        self.writer.add_scalar('acc/citys', city_acc, epoch)
        if not self.no_source_test:
            origin_acc = self.val(self.source_val_loader)
            self.writer.add_scalar('acc/origin', origin_acc, epoch)
        else:
            origin_acc = 0

        self.writer.flush()
        if city_acc > self.best_target_acc:
            self.best_target_acc = city_acc
            self.best_target_acc_source = origin_acc
            self.best_target_epoch = epoch
            self.save('best_city')

        if origin_acc > self.best_source_acc:
            self.best_source_acc = origin_acc
            self.best_source_acc_target = city_acc
            self.best_source_epoch = epoch
            self.save('best_origin')

    def val(self, dataset):
        self.backbone.eval()
        with torch.no_grad():
            self.nim.clear_cache()
            self.nim.set_max_len(len(dataset))
            for p, img, target in dataset:
                img, target = to_cuda(get_img_target(img, target))
                logits = self.backbone(img)[0]
                self.nim(logits, target)
        self.log('\nNormal validation : {}\n'.format(self.nim.get_acc()))
        if hasattr(dataset.dataset, 'format_class_iou'):
            self.log(dataset.dataset.format_class_iou(self.nim.get_class_acc()[0]) + '\n')
        return self.nim.get_acc()[0]

    def target_specific_val(self, loader):
        self.nim.clear_cache()
        self.nim.set_max_len(len(loader))
        # eval for dropout
        self.backbone.module.remove_dropout()
        self.backbone.module.not_track()
        for idx, (p, img, target) in enumerate(loader):
            if len(img.size()) == 5:
                B, D, C, H, W = img.size()
            else:
                B, C, H, W = img.size()
                D = 1
            img, target = to_cuda([img.reshape(B, D, C, H, W), target.reshape(B, D, 1, H, W)])
            for d in range(img.size(1)):
                img_d, target_d, = img[:, d], target[:, d]
                self.backbone.train()
                with torch.no_grad():
                    new_logits = self.backbone(img_d)[0]
                    self.nim(new_logits, target_d)

        self.backbone.module.recover_dropout()
        self.log('\nTarget specific validation : {}\n'.format(self.nim.get_acc()))
        if hasattr(loader.dataset, 'format_class_iou'):
            self.log(loader.dataset.format_class_iou(self.nim.get_class_acc()[0]) + '\n')
        return self.nim.get_acc()[0]

    def predict_target(self, load_path='best_city', color=False, train=False, output_path='predictions'):
        self.load(load_path)
        import skimage.io as skio
        dataset = self.target_test_loader

        output_path = Path(self.save_path / output_path)
        output_path.mkdir(exist_ok=True)

        if train:
            self.backbone.module.remove_dropout()
            self.backbone.train()
        else:
            self.backbone.eval()

        with torch.no_grad():
            self.nim.clear_cache()
            self.nim.set_max_len(len(dataset))
            for names, img, target in tqdm(dataset):
                img = to_cuda(img)
                logits = self.backbone(img)[0]
                logits = F.interpolate(logits, img.size()[2:], mode='bilinear', align_corners=True)
                preds = get_prediction(logits).cpu().numpy()
                if color:
                    trainId_preds = preds
                else:
                    trainId_preds = dataset.dataset.predict(preds)

                for pred, name in zip(trainId_preds, names):
                    file_name = name.split('/')[-1]
                    if color:
                        pred = class_map_2_color_map(pred).transpose(1, 2, 0).astype(np.uint8)
                    skio.imsave(str(output_path / file_name), pred)

    def get_string(self, epoch, it, loss_meters, acc_meters, lr, meta):
        string = '\repoch {:4}, iter : {:4}, '.format(epoch, it)
        for k, v in loss_meters.items():
            string += k + ' : {:.4f}, '.format(v.avg)
        for k, v in acc_meters.items():
            string += k + ' : {:.4f}, '.format(v.avg)

        string += 'lr : {:.6f}, meta : {}'.format(lr, meta)
        return string

    def log(self, strs):
        self.logger.info(strs)

    def print(self, strs, **kwargs):
        print(strs, **kwargs)

    def tfb_log(self, epoch, it, losses, acc):
        iteration = epoch * len(self.train_loader) + it
        for k, v in losses.items():
            self.writer.add_scalar('loss/' + k, v.val, iteration)
        for k, v in acc.items():
            self.writer.add_scalar('acc/' + k, v.val, iteration)

    def save(self, name='ckpt'):
        info = [self.best_source_acc, self.best_source_acc_target, self.best_source_epoch,
                self.best_target_acc, self.best_target_acc_source, self.best_target_epoch]
        dicts = {
            'backbone': self.backbone.state_dict(),
            'opt': self.opt_old.state_dict(),
            'epoch': self.epoch + 1,
            'best': self.best_target_acc,
            'info': info
        }
        self.print('Saving epoch : {}'.format(self.epoch))
        torch.save(dicts, self.save_path / '{}.pth'.format(name))

    def load(self, path=None, strict=False):
        if path is None:
            path = self.save_path / 'ckpt.pth'
        else:
            if 'pth' in path:
                path = path
            else:
                path = self.save_path / '{}.pth'.format(path)

        try:
            dicts = torch.load(path, map_location='cpu')
            msg = self.backbone.load_state_dict(dicts['backbone'], strict=strict)
            self.print(msg)
            if 'opt' in dicts:
                self.opt_old.load_state_dict(dicts['opt'])
            if 'epoch' in dicts:
                self.epoch = dicts['epoch']
            else:
                self.epoch = 1
            if 'best' in dicts:
                self.best_target_acc = dicts['best']
            if 'info' in dicts:
                self.best_source_acc, self.best_source_acc_target, self.best_source_epoch, \
                self.best_target_acc, self.best_target_acc_source, self.best_target_epoch = dicts['info']
            self.log('Loaded from {}, next epoch : {}, best_target : {}, best_epoch : {}\n'
                     .format(str(path), self.epoch, self.best_target_acc, self.best_target_epoch))
            return True
        except Exception as e:
            self.print(e)
            self.log('No ckpt found in {}\n'.format(str(path)))
            self.epoch = 1
            return False


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,0,1,2'
    framework = MetaFrameWork(name='exp', train_num=1, source='GSIM', target='C', debug=False, resume=True)
    framework.do_train()
    framework.val(framework.target_test_loader)
    from eval import test_one_run
    test_one_run(framework, 'previous_exps/dg_all', targets='C', batches=[16, 8, 1], normal_eval=False)
    framework.predict_target()
