import os
import json
import argparse
import cv2
import torch
import dataloaders
import models
from pathlib import Path
from tqdm import tqdm
import matplotlib.pylab as plt
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
from models.cenet_origin import CE_Net_
from utils import transforms as local_transforms
from utils.helpers import colorize_mask
from utils.metrics import *
from pathlib import Path
import time
import logging


batch_time = AverageMeter()
data_time = AverageMeter()
total_loss = AverageMeter()
total_inter, total_union = 0, 0
total_correct, total_label = 0, 0

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings=np.zeros((35,1))

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def _update_seg_metrics(correct, labeled, inter, union):
    global total_correct, total_label, total_inter, total_union
    total_correct += correct
    total_label += labeled
    total_inter += inter
    total_union += union

def _get_seg_metrics():
    global total_correct, total_label, total_inter, total_union

    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    
    mIoU = IoU.mean()
    return {
        "Pixel_Accuracy": np.round(pixAcc, 3),
        "Mean_IoU": np.round(mIoU, 3), 
        "Class_IoU": dict(zip(range(3), np.round(IoU, 3)))
    }



def main(config, resume):
    device = config['device']
    # DATA LOADERS数据加载
    test_loader = get_instance(dataloaders, 'test_loader', config)

    restore_transform = transforms.Compose([
            local_transforms.DeNormalize(test_loader.MEAN, test_loader.STD),
            transforms.ToPILImage()])
    viz_transform = transforms.Compose([
        # transforms.Resize((500, 500)),
        transforms.ToTensor()])

    # MODEL
    model = get_instance(models, 'arch', config, test_loader.dataset.num_classes)
    model.load_state_dict(torch.load(resume, map_location=device)["state_dict"])
   

    tbar = tqdm(test_loader, ncols=130)
    val_visual = []
    val_id = []
    model.to(device)
    sum_iou=0
    with torch.no_grad():
        model.eval()
        rep = 0
        for batch_idx, samples in enumerate(tbar):
            data = samples['image']
            target = samples['label']
            data_id = samples.get('image_id', None)
            edge_binary = samples.get('edge_binary', None)
            data, target = data.to(device), target.to(device)
            # b,h,w = target.shape
            torch.cuda.synchronize()
            time_start = time.time()
            # output = model(data)
            output = model(data)

          
            torch.cuda.synchronize()
            time_end = time.time()
            time_sum = time_end - time_start
            print(time_sum)

            # starter.record()
            # 
            # ender.record()
            # # WAIT FOR GPU SYNC
            # torch.cuda.synchronize()
            # curr_time = starter.elapsed_time(ender)
            timings[rep] = time_sum
            rep = rep+1
              

            if len(val_visual) < 9999:
                target_np = target.data.cpu().numpy()
                output_np = output.data.max(1)[1].cpu().numpy()
                val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])
                val_id.append(data_id)
            
            seg_metrics = eval_metrics(output, target, 2)
            _update_seg_metrics(*seg_metrics)
            # PRINT INFO
            pixAcc, mIoU, _ = _get_seg_metrics().values()
            tbar.set_description('EVAL ({}), PixelAcc: {:.3f}, Mean IoU: {:.3f} |'.format(0,pixAcc, mIoU))
    print(np.sum(timings))
    mean_syn = np.sum(timings) / 35
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
    print(mean_syn)
    savedir = Path(resume).parent / "predicted"
    if not savedir.exists():
        savedir.mkdir(parents=True, exist_ok=True)

    palette = test_loader.dataset.palette
    for data_id, (d, t, o) in zip(val_id, val_visual):
        val_img = []
        # d = restore_transform(d)
        t, o = colorize_mask(t, palette), colorize_mask(o, palette)
        # d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
        t, o = t.convert('RGB'), o.convert('RGB')

        # [d, t, o] = [viz_transform(x) for x in [d, t, o]]
        # val_img.extend([d, t, o])
        # val_img = torch.stack(val_img, 0)
        # val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
        # val_img = (val_img.permute(dims=(1, 2, 0)).data.numpy() * 255).astype(np.uint8)

        # [t, o] = [viz_transform(x) for x in [t, o]]
        # val_img.extend([t, o])
        # val_img = torch.stack(val_img, 0)
        # val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
        # val_img = (val_img.permute(dims=(1, 2, 0)).data.numpy() * 255).astype(np.uint8)
        
        o = viz_transform(o)
        val_img = o
        val_img = (val_img.permute(dims=(1, 2, 0)).data.numpy() * 255).astype(np.uint8)

        val_img = cv2.cvtColor(val_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite((savedir / f"{data_id}_mask.png").as_posix(), val_img)

class Args:
    def __init__(self, config, resume, device):
        self.config = config
        self.resume = resume
        self.device = device

if __name__ == '__main__':
    args = Args(
        config="/home/zhangsf/code/saved/marblenet/03-14_10-19/config.json",
        resume="/home/zhangsf/code/saved/marblenet/03-14_10-19/checkpoint-epoch355.pth",
        device="0,1,2,3",
    )

    config = json.load(open(args.config))
    # if args.resume:
    #     config = torch.load(args.resume)['config']
    # if args.device:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config['device'] = "cuda:3"
    config['test_loader']['args']['shuffle'] = False # 关闭乱序
    config['test_loader']['args']['batch_size'] = 1
    config['test_loader']['args']['split'] = 'test_2' # 预测训练数据
    
    main(config, args.resume)