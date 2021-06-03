import torchvision

from datamodules import *


class Cifar10Datamodule(DataModule):
    def __init__(self,
                 save_path=None,
                 train_batch_size=256,
                 test_batch_size=512,
                 valid_size=None,
                 n_workers=12):
        self._save_path = save_path
        train_transforms = self.build_train_transform()
        train_dataset = torchvision.datasets.CIFAR10(root=save_path,
                                                     train=True,
                                                     download=True,
                                                     transform=train_transforms)

        if valid_size is not None:
            if isinstance(valid_size, float):
                valid_size = int(valid_size * len(train_dataset))
            else:
                assert isinstance(valid_size, int), 'invalid valid_size: {}'.format(valid_size)
            train_indices, valid_indices = self.random_sample_valid_set(
                [cls for cls in train_dataset.targets], valid_size, self.n_classes
            )

            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

            valid_dataset = torchvision.datasets.CIFAR10(
                root=save_path,
                train=False,
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    self.normalize
                ]))

            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                sampler=train_sampler,
                num_workers=n_workers,
                pin_memory=True
            )
            self.valid = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                sampler=valid_sampler,
                num_workers=n_workers,
                pin_memory=True
            )
        else:
            self.train = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=n_workers,
                pin_memory=True
            )
            self.valid = None

        self.test = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root=save_path,
                train=False,
                download=True,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    self.normalize
                ])),
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True
        )
        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'cifar10'

    @property
    def data_shape(self):
        return 3, 32, 32

    @property
    def n_classes(self):
        return 10

    def build_train_transform(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            self.normalize
        ])

    @staticmethod
    def normalize():
        return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
