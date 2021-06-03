from configs.run_config import RunConfig, CIFAR10RunConfig


def test_run_config():
    n_epochs = 0
    init_lr = 0.05
    lr_schedule_type = 'cosine'
    lr_schedule_param = None
    dataset = None
    train_batch_size = 256
    test_batch_size = 256
    valid_size = 256
    opt_type = 'adam'
    opt_param = {'momentum': 0.9, 'nesterov': True}
    weight_decay = 0.9
    label_smoothing = 0.1
    no_decay_keys = 'bn'
    model_init = 'he_fout'
    init_div_groups = False
    validation_frequency = 1
    print_frequency = 10

    rc = RunConfig(
        n_epochs=n_epochs,
        init_lr=init_lr,
        lr_schedule_type=lr_schedule_type,
        lr_schedule_param=lr_schedule_param,
        dataset=dataset,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        valid_size=valid_size,
        opt_type=opt_type,
        opt_param=opt_param,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
        no_decay_keys=no_decay_keys,
        model_init=model_init,
        init_div_groups=init_div_groups,
        validation_frequency=validation_frequency,
        print_frequency=print_frequency
    )


def test_run_cifar10():
    crc = CIFAR10RunConfig()
    print(crc.data_config)
