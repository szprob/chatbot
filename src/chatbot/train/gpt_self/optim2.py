import numpy as np
from torch.optim import Adam


class Optimizer:
    """A wrapper class for optimizer"""

    def __init__(
        self,
        parameters,
        lr=1e-4,
        init_lr=0.03,
        betas=(0.9, 0.999),
        weight_decay: float = 0.003,
        n_warmup_steps=1000,
    ):
        self.lr = lr
        self.optim = Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        # self.init_lr = np.power(d_model, -0.5)
        self.init_lr = init_lr

    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self.optim.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optim.zero_grad()

    def _get_lr_scale(self):
        return np.min(
            [
                np.power(self.n_current_steps, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.n_current_steps,
            ]
        )

    def _update_learning_rate(self):
        """Learning rate scheduling per step"""

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self.optim.param_groups:
            param_group["lr"] = lr
