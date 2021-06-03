import math

import torch.optim


class ArchSearchConfig(object):
    def __init__(self,
                 arch_init_type: str,
                 arch_init_ratio: float,
                 arch_opt_type: str,
                 arch_lr: float,
                 arch_opt_param,
                 arch_weight_decay: float,
                 target_hardware: str,
                 ref_value):
        # Architecture parameters initialization & optimization
        self.arch_init_type = arch_init_type
        self.arch_init_ratio = arch_init_ratio

        self.opt_type = arch_opt_type
        self.lr = arch_lr
        self.opt_param = {} if arch_opt_param is None else arch_opt_param
        self.weight_decay = arch_weight_decay
        self.target_hardware = target_hardware
        self.ref_value = ref_value

    @property
    def config(self):
        config = {
            'type': type(self),
        }
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def get_update_schedule(self, num_batch):
        raise NotImplementedError

    def build_optimizer(self, params):
        if self.opt_type == 'adam':
            return torch.optim.Adam(
                params=params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                **self.opt_param
            )
        else:
            raise NotImplementedError


class GradientArchSearchConfig(ArchSearchConfig):
    def __init__(self,
                 arch_init_type: str = 'normal',
                 arch_init_ratio: float = 1e-3,
                 arch_opt_type: str = 'adam',
                 arch_lr: float = 1e-3,
                 arch_opt_param=None,
                 arch_weight_decay: float = 0.,
                 target_hardware: str = None,
                 ref_value=None,
                 grad_update_arch_param_every: int = 1,
                 grad_update_steps: int = 1,
                 grad_binary_mode: str = 'full',
                 grad_data_batch=None,
                 grad_reg_loss_type: str = None,
                 grad_reg_loss_params: dict = None,
                 **kwargs):
        super(GradientArchSearchConfig, self).__init__(
            arch_init_type=arch_init_type,
            arch_init_ratio=arch_init_ratio,
            arch_opt_type=arch_opt_type,
            arch_lr=arch_lr,
            arch_opt_param=arch_opt_param,
            arch_weight_decay=arch_weight_decay,
            target_hardware=target_hardware,
            ref_value=ref_value,
        )

        self.update_arch_param_every = grad_update_arch_param_every
        self.update_steps = grad_update_steps
        self.binary_mode = grad_binary_mode
        self.data_batch = grad_data_batch

        self.reg_loss_type = grad_reg_loss_type
        self.reg_loss_params = {} if grad_reg_loss_params is None else grad_reg_loss_params

        print(kwargs.keys())

    def get_update_schedule(self, num_batch):
        schedule = {}
        for i in range(num_batch):
            if (i + 1) % self.update_arch_param_every == 0:
                schedule[i] = self.update_steps
        return schedule

    def add_regularization_loss(self,
                                ce_loss,
                                expected_value):
        if expected_value is None:
            return ce_loss

        if self.reg_loss_type == 'mul#log':
            alpha = self.reg_loss_params.get('alpha', 1)
            beta = self.reg_loss_params.get('beta', 0.6)

            reg_loss = (torch.log(expected_value) / math.log(self.ref_value)) ** beta
            return alpha * ce_loss * reg_loss

        elif self.reg_loss_type == 'add#linear':
            reg_lambda = self.reg_loss_params.get('lambda', 2e-1)
            reg_loss = reg_lambda * (expected_value - self.ref_value) / self.ref_value
            return ce_loss + reg_loss

        elif self.reg_loss_type is None:
            return ce_loss

        else:
            raise ValueError('Do not support: {}'.format(self.reg_loss_type))

# TODO(joohyun): RL configuration
