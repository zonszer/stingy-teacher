"""
   CIFAR-10 CIFAR-100, Tiny-ImageNet data loader
"""
import random
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets


class inv_transform(object):
    def __call__(self, t):
        """
        Args: t (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns: Tensor: Normalized image.
        """
        from albumentations.augmentations import Normalize, FromFloat, Compose
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.247, 0.243, 0.261]
        train_invtransformer = Compose(
        [
            Normalize(
                mean=tuple(-m / s for m, s in zip(mean, std)),
                std=tuple(1.0 / s for s in std),
                max_pixel_value=1.0,
            ),
            FromFloat(max_value=255, dtype="uint8"),
        ])
        return train_invtransformer(t)

    def __repr__(self):
        return self.__class__.__name__+'()'

def generate_path_replace(input_path):
    # Extract the directory and file name from the input path
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    # Generate the output path by replacing "x_train" with "y_train" in the filename
    output_filename = filename.replace("x_train", "y_train")
    output_path = os.path.join(directory, output_filename)
    return output_path


class Cifar10Dataset(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target



class Cifar10Dataset_img(Dataset):
    def __init__(self, x_path=None, y_path=None, 
                 x_data=None, y_data=None,
                 transform=None):
        super().__init__()
        if x_path != None:
            if y_path == None:
                y_path = generate_path_replace(x_path)
            try: 
                self.data = np.load(x_path)
                self.targets = np.load(y_path)
            except:
                self.data = torch.load(x_path, map_location="cpu")
                self.targets = torch.load(y_path, map_location="cpu")
            self.transform = transform
            assert self.data.shape[0] == self.targets.shape[0]
        elif x_data != None and y_data != None:
            self.data = x_data
            self.targets = y_data
            self.transform = transform
        else:
            raise ValueError("x_path and x_data can't be None at the same time")
    
    def spilit_dataset(self, spilit_ratio):
        spilit_size = int(spilit_ratio*len(self.data))
        self.data, data_remain = self.data[:spilit_size], self.data[spilit_size:]
        self.targets, targets_remain = self.targets[:spilit_size], self.targets[spilit_size:]

        dev_dataset = Cifar10Dataset_img(x_data=data_remain, y_data=targets_remain, 
                                        transform=self.transform)
        return self, dev_dataset

    
    def __getitem__(self, index):
        img = self.data[index]
        y = self.targets[index]
        img = Image.fromarray(img)
        if self.transform:
            # Converts the data from numpy to torch tensor
            img = self.transform(img)
        return img, y

    def __len__(self):
        return len(self.data)


def fetch_dataloader(mode='clean_data', params=None):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    # using random crops and horizontal flip for train set
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.247, 0.243, 0.261]
    CIFAR10_path = './data/data-cifar10'
    CIFAR100_path = './data/data-cifar100'
    if params.augmentation:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    if mode == 'clean_data':
        # ************************************************************************************
        if params.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root=CIFAR10_path,
                                                    train=True,
                                                    download=True, transform=train_transformer)
            devset = torchvision.datasets.CIFAR10(root=CIFAR10_path, train=False,
                                                download=True, transform=dev_transformer)
        
        # ************************************************************************************
        elif params.dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(root=CIFAR100_path, train=True,
                                                    download=True, transform=train_transformer)
            devset = torchvision.datasets.CIFAR100(root=CIFAR100_path,
                                                train=False,
                                                download=True, transform=dev_transformer)

        # ************************************************************************************
        elif params.dataset == 'tiny_imagenet':
            mean = [0.4802, 0.4481, 0.3975]
            std = [0.2302, 0.2265, 0.2262]
            data_dir = './data/tiny-imagenet-200/'
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            }
            train_dir = data_dir + 'train/'
            test_dir = data_dir + 'val/'
            trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
            devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])
    else:
        assert mode == 'posion_data'
        assert params.dataset == 'cifar10'
        # train_transformer = transforms.Compose([
        #     transforms.Normalize(mean, std)])
        trainset = Cifar10Dataset_img(x_path=params.pData_path, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root=CIFAR10_path,
                                              train=False,
                                              download=True, transform=dev_transformer)

    if hasattr(params, 'use_entire_dataset'):
        params.batch_size = len(trainset)

    invtransformer = transforms.Compose([
        transforms.Normalize(
            mean=tuple(-m / s for m, s in zip(mean, std)),
            std=tuple(1.0 / s for s in std),
        ),
    ])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
                                              shuffle=True, num_workers=params.num_workers)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
                                            shuffle=False, num_workers=params.num_workers)
    trainloader.dataset.invtransformer = invtransformer
    devloader.dataset.invtransformer = invtransformer
    return trainloader, devloader

def fetch_subset_dataloader_(mode='clean_data', params=None):
    """
    Fetch subset dataloader with hyperparameters (default no data aug and norm)
    """
    # using random crops and horizontal flip for train set
    # mean = [0.4914, 0.4822, 0.4465]
    # std = [0.247, 0.243, 0.261]
    CIFAR10_path = './data/data-cifar10'
    CIFAR100_path = './data/data-cifar100'
    if params.augmentation:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)       #remove std norm
            ])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(), 
        # transforms.Normalize(mean, std)
        ])
    if mode == 'clean_data':
        # ************************************************************************************
        if params.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root=CIFAR10_path,
                                                    train=True,
                                                    download=True, transform=train_transformer)
            devset = torchvision.datasets.CIFAR10(root=CIFAR10_path, train=False,
                                                download=True, transform=dev_transformer)
        
        # ************************************************************************************
        elif params.dataset == 'cifar100':
            trainset = torchvision.datasets.CIFAR100(root=CIFAR100_path, train=True,
                                                    download=True, transform=train_transformer)
            devset = torchvision.datasets.CIFAR100(root=CIFAR100_path,
                                                train=False,
                                                download=True, transform=dev_transformer)

        # ************************************************************************************
        elif params.dataset == 'tiny_imagenet':
            mean = [0.4802, 0.4481, 0.3975]
            std = [0.2302, 0.2265, 0.2262]
            data_dir = './data/tiny-imagenet-200/'
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomRotation(20),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ])
            }
            train_dir = data_dir + 'train/'
            test_dir = data_dir + 'val/'
            trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
            devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])
    else:
        assert mode == 'posion_data'
        assert params.dataset == 'cifar10'
        # train_transformer = transforms.Compose([
        #     transforms.Normalize(mean, std)])
        trainset = Cifar10Dataset_img(x_path=params.pData_path, transform=train_transformer)
        trainset, devset = trainset.spilit_dataset(spilit_ratio=0.8)    #dev transform==train transform

    if hasattr(params, 'use_entire_dataset'):
        params.batch_size = len(trainset)

    invtransformer = transforms.Compose([
        transforms.Normalize(
            mean=tuple(-m / s for m, s in zip(mean, std)),
            std=tuple(1.0 / s for s in std),
        ),
    ])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
                                              shuffle=True, num_workers=params.num_workers)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
                                            shuffle=False, num_workers=params.num_workers)
    trainloader.dataset.invtransformer = invtransformer
    devloader.dataset.invtransformer = invtransformer
    return trainloader, devloader


def fetch_subset_dataloader(types, params):
    """
    Use only a subset of dataset for KD training, depending on params.subset_percent
    """

    # using random crops and horizontal flip for train set
    if params.augmentation:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    # ************************************************************************************
    if params.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='/home/dayong/CV/registration/ZJH/stingy-teacher/data/data-cifar10', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR10(root='/home/dayong/CV/registration/ZJH/stingy-teacher/data/data-cifar10', train=False,
                                              download=True, transform=dev_transformer)

    # ************************************************************************************
    elif params.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='/home/dayong/CV/registration/ZJH/stingy-teacher/data/data-cifar100', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR100(root='/home/dayong/CV/registration/ZJH/stingy-teacher/data/data-cifar100', train=False,
                                              download=True, transform=dev_transformer)

    # ************************************************************************************
    elif params.dataset == 'tiny_imagenet':
        data_dir = './data/tiny-imagenet-200/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
        }
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'val/'
        trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
        devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])

    trainset_size = len(trainset)
    indices = list(range(trainset_size))
    split = int(np.floor(params.subset_percent * trainset_size))
    np.random.seed(230)
    np.random.shuffle(indices)
    train_sampler = SubsetRandomSampler(indices[:split])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
        sampler=train_sampler, num_workers=params.num_workers, pin_memory=True)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
        shuffle=False, num_workers=params.num_workers, pin_memory=True)

    if types == 'train':
        dl = trainloader
    else:
        dl = devloader

    return dl
