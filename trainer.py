import imp
import torch
import torch.nn as nn
import time
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from utils import transforms as local_transforms
from base import BaseTrainer, DataPrefetcher
from utils.helpers import colorize_mask
from utils.metrics import eval_metrics, AverageMeter
from tqdm import tqdm
from PIL import Image
import cv2

class Trainer(BaseTrainer):
    def __init__(self, model, loss, resume, config, train_loader, val_loader=None, test_loader=None, train_logger=None, prefetch=True):
        super(Trainer, self).__init__(model, loss, resume, config, train_loader, val_loader, test_loader, train_logger)
        
        self.wrt_mode, self.wrt_step = 'train_', 0
        self.log_step = config['trainer'].get('log_per_iter', int(np.sqrt(self.train_loader.batch_size)))
        if config['trainer']['log_per_iter']: self.log_step = int(self.log_step / self.train_loader.batch_size) + 1

        self.num_classes = self.train_loader.dataset.num_classes

        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])
        self.viz_transform = transforms.Compose([
            transforms.Resize((400, 400)),
            transforms.ToTensor()])
        
        if self.device == torch.device('cpu'): prefetch = False
        if prefetch:
            self.train_loader = DataPrefetcher(train_loader, device=self.device)
            self.val_loader = DataPrefetcher(val_loader, device=self.device)
            self.test_loader = DataPrefetcher(test_loader, device=self.device)

        torch.backends.cudnn.benchmark = True

    def _train_epoch(self, epoch):
        self.logger.info('\n')
            
        self.model.train()
        if self.config['arch']['args']['freeze_bn']:
            if isinstance(self.model, torch.nn.DataParallel): self.model.module.freeze_bn()
            else: self.model.freeze_bn()
        self.wrt_mode = 'train'

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=130)
        train_visual = []
        for batch_idx, samples in enumerate(tbar):
            data = samples['image']
            target = samples['label']
            data_id = samples.get('image_id', None)
            edge_binary = samples.get('edge_binary', None)


            self.data_time.update(time.time() - tic)
            data, target = data.to(self.device), target.to(self.device)

            # LOSS & OPTIMIZE
            self.optimizer.zero_grad()
            
            output = self.model(data)#8,2,200,200
            # output = rebuild_images(output,(400,800),(200,200))
            if self.config['arch']['type'][:3] == 'PSP' and self.config['arch']['args'].get('use_aux', False):
                # use_aux使用分支进行分类
                assert output[0].size()[2:] == target.size()[1:]
                assert output[0].size()[1] == self.num_classes
                loss = self.loss(output[0], target)
                loss += self.loss(output[1], target) * 0.4
                output = output[0]
            # elif self.config['arch']['type'] == 'DANet':
            #     assert output.size()[1] == self.num_classes
            #     output = nn.UpsamplingBilinear2d((target.size()[1], target.size()[2]))(output)
            #     loss = self.loss(output, target)
            elif self.config['arch']['type'][len('Decoupled'):] == 'Decoupled':
                # body、edge解耦模块
                seg_final, seg_body, seg_edge = output
                assert seg_final.size()[2:] == seg_final.size()[1:]
                assert seg_final.size()[1] == self.num_classes
                

            else:
                assert output.size()[2:] == target.size()[1:]
                assert output.size()[1] == self.num_classes
                loss = self.loss(output, target)

            if isinstance(self.loss, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())

            # measure elapsed time
            self.batch_time.update(time.time() - tic)
            tic = time.time()

            # LOGGING & TENSORBOARD
            if batch_idx % self.log_step == 0:
                self.wrt_step = (epoch - 1) * len(self.train_loader) + batch_idx
                self.writer.add_scalar(f'{self.wrt_mode}/loss', loss.item(), self.wrt_step)

            # FOR EVAL
            seg_metrics = eval_metrics(output, target, self.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()

            # LIST OF IMAGE TO VIZ (15 images)
            if len(train_visual) < 15:
                target_np = target.data.cpu().numpy()
                output_np = output.data.max(1)[1].cpu().numpy()
                train_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])
            
            # PRINT INFO
            tbar.set_description('TRAIN ({}) | Loss: {:.3f} | Acc {:.3f} mIoU {:.3f} | B {:.2f} D {:.2f} lr {:.6f}|'.format(
                                                epoch, self.total_loss.average, 
                                                pixAcc, mIoU,
                                                self.batch_time.average, self.data_time.average, self.lr_scheduler.get_last_lr()[0]))
        # 学习率更新
        self.lr_scheduler.step()
        # METRICS TO TENSORBOARD
        seg_metrics = self._get_train_seg_metrics()
        for k, v in list(seg_metrics.items())[:-1]: 
            self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
        # 记录每个类的mIOU
        for k, v in list(seg_metrics.items())[-1:]:
            for class_, mIOU in v.items():
                self.writer.add_scalar(f'{self.wrt_mode}/{k}/{class_}', mIOU, self.wrt_step)

        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f'{self.wrt_mode}/Learning_rate_{i}', opt_group['lr'], self.wrt_step)
            #self.writer.add_scalar(f'{self.wrt_mode}/Momentum_{k}', opt_group['momentum'], self.wrt_step)
        
        # WRTING & VISUALIZING THE MASKS
        train_img = []
        palette = self.train_loader.dataset.palette
        for d, t, o in train_visual:
            d = self.restore_transform(d)
            t, o = colorize_mask(t, palette), colorize_mask(o, palette)
            d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
            [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
            train_img.extend([d, t, o])
        train_img = torch.stack(train_img, 0)
        train_img = make_grid(train_img.cpu(), nrow=3, padding=5)
        self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', train_img, self.wrt_step)

        # RETURN LOSS & METRICS
        log = {'loss': self.total_loss.average,
                **seg_metrics}

        #if self.lr_scheduler is not None: self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        if self.val_loader is None:
            self.logger.warning('Not data loader was passed for the validation step, No validation is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, samples in enumerate(tbar):
                data = samples['image']
                target = samples['label']
                data_id = samples.get('image_id', None)
                edge_binary = samples.get('edge_binary', None)
                data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)

                # b,h,w = target.shape

                # output = rebuild_images(output,(h,w),(200,200))
                if self.config['arch']['type'][:3] == 'PSP' and self.config['arch']['args'].get('use_aux', False):
                    # use_aux使用分支进行分类
                    assert output[0].size()[2:] == target.size()[1:]
                    assert output[0].size()[1] == self.num_classes
                    loss = self.loss(output[0], target)
                    loss += self.loss(output[1], target) * 0.4
                    output = output[0]
                # elif self.config['arch']['type'] == 'DANet':
                #     assert output.size()[1] == self.num_classes
                #     output = nn.UpsamplingBilinear2d((target.size()[1], target.size()[2]))(output)
                #     loss = self.loss(output, target)
                elif self.config['arch']['type'][len('Decoupled'):] == 'Decoupled':
                    # body、edge解耦模块
                    seg_final, seg_body, seg_edge = output
                    assert seg_final.size()[2:] == seg_final.size()[1:]
                    assert seg_final.size()[1] == self.num_classes
                

                else:
                    assert output.size()[2:] == target.size()[1:]
                    assert output.size()[1] == self.num_classes
                    loss = self.loss(output, target)
                
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.3f}, Mean IoU: {:.3f} |'.format( epoch,
                                                self.total_loss.average,
                                                pixAcc, mIoU))

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
            # 记录每个类的mIOU
            for k, v in list(seg_metrics.items())[-1:]:
                for class_, mIOU in v.items():
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}/{class_}', mIOU, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log
    
    def _test_epoch(self, epoch):
        if self.test_loader is None:
            self.logger.warning('Not data loader was passed for the test step, No test is performed !')
            return {}
        self.logger.info('\n###### EVALUATION ######')

        self.model.eval()
        self.wrt_mode = 'test'

        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=130)
        with torch.no_grad():
            test_visual = []
            for batch_idx, samples in enumerate(tbar):
                data = samples['image']
                target = samples['label']
                data_id = samples.get('image_id', None)
                edge_binary = samples.get('edge_binary', None)
                data, target = data.to(self.device), target.to(self.device)
                # LOSS
                output = self.model(data)

                # b,h,w = target.shape

                # output = rebuild_images(output,(h,w),(200,200))
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(test_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    test_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])

                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.3f}, Mean IoU: {:.3f} |'.format( epoch,
                                                self.total_loss.average,
                                                pixAcc, mIoU))

            # WRTING & VISUALIZING THE MASKS
            test_img = []
            palette = self.train_loader.dataset.palette
            for d, t, o in test_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                test_img.extend([d, t, o])
            test_img = torch.stack(test_img, 0)
            test_img = make_grid(test_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', test_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.test_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics()
            for k, v in list(seg_metrics.items())[:-1]: 
                self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)
            
            # 记录每个类的mIOU
            for k, v in list(seg_metrics.items())[-1:]:
                for class_, mIOU in v.items():
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}/{class_}', mIOU, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

 

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }
    
    def _get_train_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "train_PA": np.round(pixAcc, 3),
            "train_Mean_IoU": np.round(mIoU, 3),
            "train_Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }