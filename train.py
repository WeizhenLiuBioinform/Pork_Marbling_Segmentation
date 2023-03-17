from cv2 import split
import os
import json
import argparse
import torch
import torch.optim
import dataloaders
import models
import inspect
import math
from time import strftime, localtime
from models.coanet import CoANet
from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer import Trainer
from pathlib import Path
import logging



# from PIL import ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    now = strftime('%m-%d_%H-%M',localtime())
    logfile = Path(f"{config['loggedir']}/{config['name']}/{now}/run.log")
    if not logfile.parent.exists():
        logfile.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=logfile.as_posix(), level=logging.INFO, format='')

    train_logger = Logger()

    # DATA LOADERS数据加载
    train_loader = get_instance(dataloaders, 'train_loader', config)
    val_loader = get_instance(dataloaders, 'val_loader', config)
    test_loader = get_instance(dataloaders, 'test_loader', config)

    # MODEL
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    # model = CoANet()
    print(f'\n{model}\n')


    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index = config['ignore_index'])


    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_logger=train_logger,
        prefetch=config['prefetch']
    )

    trainer.train()

class Args:
    def __init__(self, config, resume, device):
        self.config = config
        self.resume = resume
        self.device = device

if __name__=='__main__':
  
    args = Args(
        config="/home/zhangsf/code_prepare/config.json",
        resume="",
        device="0,1,2,3",
    )

    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    config['test_loader']['args']['split']='val_2'
    config['train_loader']['args']['split']='train_2'
    config['val_loader']['args']['split']='val_2'
    main(config, args.resume)

  