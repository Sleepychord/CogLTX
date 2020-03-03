import torch
from torch.optim.lr_scheduler import _LRScheduler
class WarmupLinearLR(_LRScheduler):
    def __init__(self, optimizer, step_size, peak_percentage=0.1, min_lr=1e-5, last_epoch=-1):
        self.step_size = step_size
        self.peak_step = peak_percentage * step_size
        self.min_lr = min_lr
        super(WarmupLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ret = []
        for base_lr in self.base_lrs:
            if self._step_count <= self.peak_step:
                ret.append(self.min_lr + (base_lr - self.min_lr) * self._step_count / self.peak_step)
            else:
                ret.append(self.min_lr + max(0, (base_lr - self.min_lr) * (self.step_size - self._step_count) / (self.step_size - self.peak_step)))
        return ret

def mocofy(optimizer, son_params, momentum=0.9):
    assert isinstance(optimizer, torch.optim.Optimizer)
    optimizer.son_params = son_params
    optimizer.moco_momentum = momentum
    old_step = optimizer_self.step
    def new_step(optimizer_self, *args):
        old_step(optimizer_self, *args)
        for p1, p2 in zip(optimizer_self.parameters(), son_params):
            p2.data.mul_(momentum).add_(1-momentum, p1.detach().data)
    optimizer.step = new_step
    return optimizer

