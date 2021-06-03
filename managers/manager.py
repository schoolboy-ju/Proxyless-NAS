import copy
import os

import torch
from torch import nn

from configs.run_config import RunConfig
from configs.nas_config import ArchSearchConfig


class RunManager(object):
    def __init__(self,
                 path: str,
                 net,
                 run_config: RunConfig,
                 out_log: bool = True,
                 measure_latency=None):
        self._path = path
        self._net = net
        self._run_config = run_config
        self._out_log = out_log
        self._logs_path, self._save_path = None, None
        self._best_acc = 0
        self._start_epoch = 0

        # Initialize model (default)
        self._net.init_model(run_config.model_init,
                             run_config.init_div_groups)

        # A copy of net on cpu for latency estimation & mobile latency model
        self._net_on_cpu_for_latency = copy.deepcopy(self._net).cpu()

        # TODO(joohyun): latency estimator?

        # Move network to GPU if available
        if torch.cuda.is_available():
            self._device = torch.device('cuda:0')
            self._net = torch.nn.DataParallel(self._net)
            self._net.to(self._device)
            torch.backends.cudnn.benchmark = True
        else:
            raise ValueError

        # Net info
        self._print_net_info(measure_latency)

        self._criterion = nn.CrossEntropyLoss()
        if self._run_config.no_decay_keys:
            keys = self._run_config.no_decay_keys.split('#')
            self._optimizer = self._run_config.build_optimizer([
                self._net.module.get_parameters(keys, mode='exclude'),  # Parameters with weight decay
                self._net.module.get_parameters(keys, mode='include'),  # Parameters without weight decay
            ])
        else:
            self._optimizer = self._run_config.build_optimizer(self._net.module.weight_parameters())

    @property
    def save_path(self):
        if self._save_path is None:
            save_path = os.path.join(self._path, 'checkpoint')


class ArchSearchRunManager(object):
    def __init__(self,
                 path: str,
                 super_net,
                 run_config: RunConfig,
                 arch_search_config: ArchSearchConfig):
        # Init weight parameters & build weight optimizer
        ...
