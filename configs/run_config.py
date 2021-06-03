class RunConfig(object):
    def __init__(self,
                 n_epochs: int,
                 init_lr: float,
                 lr_schedule_type,
                 lr_schedule_param,
                 dataset,
                 train_batch_size,
                 test_batch_size,
                 valid_size,
                 opt_type,
                 opt_param,
                 weight_decay,
                 label_smoothing,
                 no_decay_keys,
                 model_init,
                 init_div_groups: bool,
                 validation_frequency,
                 print_frequency):

        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.model_init = model_init
        self.init_div_groups = init_div_groups
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

        self._data_provider = None
        self._train_iter, self._valid_iter, self._test_iter = None, None, None

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)


class CIFAR10RunConfig(RunConfig):

    def __init__(self,
                 n_epochs=150,
                 init_lr=0.05,
                 lr_schedule_type='cosine',
                 lr_schedule_param=None,
                 dataset='cifar10',
                 train_batch_size=256,
                 test_batch_size=500,
                 valid_size=None,
                 opt_type='sgd',
                 opt_param=None,
                 weight_decay=4e-5,
                 label_smoothing=0.1,
                 no_decay_keys='bn',
                 model_init='he_fout',
                 init_div_groups=False,
                 validation_frequency=1,
                 print_frequency=10,
                 n_worker=12):
        super(CIFAR10RunConfig, self).__init__(n_epochs,
                                               init_lr,
                                               lr_schedule_type,
                                               lr_schedule_param,
                                               dataset,
                                               train_batch_size,
                                               test_batch_size,
                                               valid_size,
                                               opt_type,
                                               opt_param,
                                               weight_decay,
                                               label_smoothing,
                                               no_decay_keys,
                                               model_init,
                                               init_div_groups,
                                               validation_frequency,
                                               print_frequency)

        self.n_worker = n_worker

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
        }
