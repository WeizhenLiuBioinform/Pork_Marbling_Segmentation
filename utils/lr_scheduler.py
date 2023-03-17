import math
from sched import scheduler
from  torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, MultiStepLR, CosineAnnealingLR, ExponentialLR
from  torch.optim.lr_scheduler import OneCycleLR

def Step(optimizer, step_size=5, gamma=0.9, last_epoch=-1):
    """
    每过一个step_size个epoch，学习率乘gamma
    """
    scheduler = StepLR(optimizer, step_size, gamma, last_epoch)
    return scheduler

def MultiStep(optimizer, milestones=[50, 100, 150], gamma=0.1, last_epoch=-1):
    """
    每到达一个milestones的epoch，学习率乘gamma
    """
    scheduler = MultiStepLR(optimizer, milestones, gamma, last_epoch)
    return scheduler

def CosineAnnealing(optimizer, T_max_epoch, eta_min=1e-6, last_epoch=-1):
    """
    余弦衰减
    """
    scheduler = CosineAnnealingLR(optimizer, T_max_epoch, eta_min, last_epoch)
    return scheduler

def Exponential(optimizer, gamma=0.9, last_epoch=-1):
    """
    指数衰减，底数为gamma，次幂为epoch
    """
    scheduler = ExponentialLR(optimizer, gamma, last_epoch)
    return scheduler

def OneCycle(optimizer, epochs, iters_per_epoch=1, **kwargs):
    """
    https://blog.csdn.net/zisuina_2/article/details/103236864
    """
    scheduler = OneCycleLR(optimizer, steps_per_epoch=iters_per_epoch, epochs=epochs, **kwargs)
    return scheduler

# def LR_WarmUp(optimizer, warmup_scheduler:_LRScheduler, main_scheduler: _LRScheduler, warmup_epochs:int=10):
#     """
#     加上预热学习率
#     """
#     def lambda_lr(epoch):
#         if epoch < warmup_epochs:
#             # warm up schedule
#             return warmup_scheduler.get_lr()


# class Poly(_LRScheduler):
#     def __init__(self, optimizer, num_epochs, iters_per_epoch=0, warmup_epochs=10, last_epoch=-1):
#         self.iters_per_epoch = iters_per_epoch
#         self.cur_iter = 0
#         self.N = num_epochs * iters_per_epoch
#         self.warmup_iters = warmup_epochs * iters_per_epoch
#         super(Poly, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         T = self.last_epoch * self.iters_per_epoch + self.cur_iter
#         factor =  pow((1 - 1.0 * T / self.N), 0.9)
#         if self.warmup_iters > 0 and T < self.warmup_iters:
#             factor = 1.0 * T / self.warmup_iters

#         self.cur_iter %= self.iters_per_epoch
#         self.cur_iter += 1
#         return [base_lr * factor for base_lr in self.base_lrs]


# class Poly(_LRScheduler):
#     def __init__(self, optimizer, num_epochs, iters_per_epoch=0, warmup_epochs=0, last_epoch=-1):
#         self.iters_per_epoch = iters_per_epoch
#         self.cur_iter = 0
#         self.N = num_epochs * iters_per_epoch
#         self.warmup_iters = warmup_epochs * iters_per_epoch
#         super(Poly, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         T = self.last_epoch * self.iters_per_epoch + self.cur_iter
#         factor =  pow((1 - 1.0 * T / self.N), 0.9)
#         if self.warmup_iters > 0 and T < self.warmup_iters:
#             factor = 1.0 * T / self.warmup_iters

#         self.cur_iter %= self.iters_per_epoch
#         self.cur_iter += 1
#         return [base_lr * factor for base_lr in self.base_lrs]

class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch=0, warmup_epochs=0, last_epoch=-1):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs
        self.warmup_iters = warmup_epochs
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch
        factor =  pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        return [base_lr * factor for base_lr in self.base_lrs]

# class OneCycle(_LRScheduler):
#     def __init__(self, optimizer, num_epochs, iters_per_epoch=0, last_epoch=-1,
#                     momentums = (0.85, 0.95), div_factor = 25, phase1=0.3):
#         self.iters_per_epoch = iters_per_epoch
#         self.cur_iter = 0
#         self.N = num_epochs * iters_per_epoch
#         self.phase1_iters = int(self.N * phase1) # warmup，学习率由低上升至最大学习率，phase1迭代次数
#         self.phase2_iters = (self.N - self.phase1_iters)# 学习率衰减
#         self.momentums = momentums
#         self.mom_diff = momentums[1] - momentums[0]

#         self.low_lrs = [opt_grp['lr']/div_factor for opt_grp in optimizer.param_groups]
#         self.final_lrs = [opt_grp['lr']/(div_factor * 1e4) for opt_grp in optimizer.param_groups]
#         super(OneCycle, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         T = self.last_epoch * self.iters_per_epoch + self.cur_iter
#         self.cur_iter %= self.iters_per_epoch
#         self.cur_iter += 1

#         # Going from base_lr / 25 -> base_lr
#         if T <= self.phase1_iters:
#             cos_anneling =  (1 + math.cos(math.pi * T / self.phase1_iters)) / 2
#             for i in range(len(self.optimizer.param_groups)):
#                 self.optimizer.param_groups[i]['momentum'] = self.momentums[0] + self.mom_diff * cos_anneling

#             return [base_lr - (base_lr - low_lr) * cos_anneling 
#                     for base_lr, low_lr in zip(self.base_lrs, self.low_lrs)]

#         # Going from base_lr -> base_lr / (25e4)
#         T -= self.phase1_iters
#         cos_anneling =  (1 + math.cos(math.pi * T / self.phase2_iters)) / 2

#         for i in range(len(self.optimizer.param_groups)):
#             self.optimizer.param_groups[i]['momentum'] = self.momentums[1] - self.mom_diff * cos_anneling
#         return [final_lr + (base_lr - final_lr) * cos_anneling 
#             for base_lr, final_lr in zip(self.base_lrs, self.final_lrs)]



if __name__ == "__main__":
    import torchvision
    import torch
    import matplotlib.pylab as plt

    resnet = torchvision.models.resnet34()
    params = {
        "lr": 0.01,
        "weight_decay": 0.001,
        "momentum": 0.9
    }
    optimizer = torch.optim.SGD(params=resnet.parameters(), **params)

    epochs = 200
    iters_per_epoch = 10
    lrs = []
    mementums = []
    # lr_scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=iters_per_epoch, epochs=epochs)
    lr_scheduler = Poly(optimizer, epochs, iters_per_epoch)

    for epoch in range(epochs):
        for i in range(iters_per_epoch):
            # lr_scheduler(optimizer, i, epoch)
            mementums.append(optimizer.param_groups[0]['momentum'])
        lr_scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    
    print(lrs[-1])
    plt.ylabel("learning rate")
    plt.xlabel("iteration")
    plt.plot(lrs)
    plt.savefig("lr.jpg")

    # plt.ylabel("momentum")
    # plt.xlabel("iteration")
    # plt.plot(mementums)
    # plt.savefig("./fig2.jpg")

    