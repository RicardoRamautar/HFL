import math
from mmcv import print_log

class LRScheduler():
    def __init__(self, 
                 policy: str,
                 total_epochs: int,
                 offset: int = 0,
                 **kwargs):
        """Learning rate scheduler

        Args:
            policy (str): Type of learning rate policy.
            total_epochs (int): Total number of epochs across federated training.
            offset (int): Epoch to start from. Default = 0.
        """
        self.total_epochs = total_epochs
        self.offset = offset
        self.total_iterations = None
        self.iters_per_epoch = None

        if policy == "cyclic":
            self.policy = CyclicScheduler(**kwargs)
        else:
            raise ValueError(f'Requested unknown learning rate policy.'
                             f'Available policies include: "cyclic", but got "{policy}".')

    def set_total_iters(self, iters_per_epoch):
        self.iters_per_epoch = iters_per_epoch
        self.total_iterations = iters_per_epoch * self.total_epochs
        self.policy.update_total_iterations(self.total_iterations)

    def set_offset(self, offset):
        self.offset = offset
        print_log(f"[LRScheduler] offset set to: {offset}", logger='root' )

    def lr_at(self, iteration):
        assert self.iters_per_epoch is not None, \
            f"iters_per_epoch unavailable."

        curr_iter = self.offset * self.iters_per_epoch + iteration
        lr = self.policy.lr_at(curr_iter)

        # print_log(f"[LRScheduler] learning rate at iteration {curr_iter} set to: {lr}", logger='root' )

        return curr_iter, lr


class CyclicScheduler():
    def __init__(self,
                 initial_lr: float,
                 min_lr: float,
                 max_lr: float,
                 pct_start: float = 0.4):

        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.pct_start = pct_start

        self.total_iters = None

    def update_total_iterations(self, total_iterations):
        self.total_iters = total_iterations

    def lr_at(self, iteration):
        assert self.total_iters is not None, "self.total_iters has not been set yet."

        if iteration < 0 or iteration > self.total_iters:
            raise ValueError(f"step {iteration} outside [0, {self.total_iters}]")

        warmup_steps = int(self.pct_start * self.total_iters)

        def anneal_cos(start, end, pct):
            return end + (start - end) / 2.0 * (math.cos(math.pi * pct) + 1)

        if iteration < warmup_steps:
            pct = iteration / warmup_steps if warmup_steps > 0 else 1.0
            return anneal_cos(self.initial_lr, self.max_lr, pct)
        else:
            pct = (iteration - warmup_steps) / (self.total_iters - warmup_steps)
            return anneal_cos(self.max_lr, self.min_lr, pct)

