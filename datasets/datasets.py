import errno
import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import torchvision
from torchvision import datasets
from torchvision import transforms

import numpy as np
from skimage import io, transform

from tqdm import tqdm
from sklearn.model_selection import train_test_split

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class Office31(Dataset):

    def __init__(self, domain, root="office31/", train=True, partial=False, transform=None, target_transform=None):
        super(Office31, self).__init__()
        self.root = os.path.join(os.getcwd(), 'datasets', root)
        self.train = train
        self.partial = partial
        self.transform = transform
        self.target_transform = target_transform

        if self.partial:
            dataset_ = np.load(os.path.join(self.root, domain+"10.npz"))
        else:
            dataset_ = np.load(os.path.join(self.root, domain+"31.npz"))

        self.data, self.label = dataset_["data"], dataset_["label"]

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        data = Image.fromarray(np.uint8(data*255.0), mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return (len(self.label))


class OfficeHome(Dataset):

    def __init__(self, domain, root="office-home/", train=True, transform=None, target_transform=None):
        super(OfficeHome, self).__init__()
        self.root = os.path.join(os.getcwd(), 'datasets', root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        dataset_ = np.load(os.path.join(self.root, domain+"65.npz"))

        self.data, self.label = dataset_["data"], dataset_["label"]

    def __getitem__(self, index):

        data, label = self.data[index], self.label[index]
        data = Image.fromarray(np.uint8(data*255.0), mode="RGB")

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return data, label

    def __len__(self):
        return (len(self.label))


class DomainNet(Dataset):
    def __init__(self, domain, args, root="datasets/",
                 transform=None, target_transform=None, test=False):
        if test:
            imgs, labels = make_dataset_fromlist(os.path.join(
                os.getcwd(), "datasets/txt", domain+"_test.txt"))
            args.num_classes = len(return_classlist(os.path.join(
                os.getcwd(), "datasets/txt", domain+"_test.txt")))
        else:
            imgs, labels = make_dataset_fromlist(os.path.join(
                os.getcwd(), "datasets/txt", domain+"_train.txt"))
            args.num_classes = len(return_classlist(os.path.join(
                os.getcwd(), "datasets/txt", domain+"_train.txt")))
        self.imgs = imgs
        self.label = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root  # READ FROM SSD

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.label[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def get_dataset_visda(args, domain, train):
    # Data loading code
    dir_s = os.path.join(os.getcwd()+"/datasets/data/VisDA-18", args.source)
    dir_t = os.path.join(os.getcwd()+"/datasets/data/VisDA-18", args.target)

    if not os.path.isdir(dir_s):
        raise ValueError(
            'The required data path is not exist, please download the dataset!')
    if not os.path.isdir(dir_t):
        raise ValueError(
            'The required data path is not exist, please download the dataset!')

    # transformation
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if domain == 'source':
        dataset = datasets.ImageFolder(root=dir_s, transform=data_transforms)
    else:
        dataset = datasets.ImageFolder(root=dir_t, transform=data_transforms)

    return dataset


def get_source_domain(source_name, args, train=True):
    # Define root folder to store source dataset
    root = CURRENT_DIR_PATH + "/source"
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # Define image source domain transformation
    source_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Define source dataset
    if source_name == 'amazon' or source_name == 'dslr' or source_name == 'webcam':
        source_dataset = Office31(
            source_name, transform=source_transforms, partial=False)
    elif source_name == 'Art' or source_name == 'Clipart' or source_name == 'Product' or source_name == 'Real World':
        source_dataset = OfficeHome(source_name, transform=source_transforms)
    elif source_name == 'real' or source_name == 'synthetic':  # visda-18
        source_dataset = get_dataset_visda(args, 'source', train)
    elif source_name == "real-DN" or source_name == "clipart" or source_name == "sketch":
        source_dataset = DomainNet(
            source_name, args, transform=source_transforms)

    loader = torch.utils.data.DataLoader(
        source_dataset, batch_size=args.batch_size, shuffle=train, num_workers=args.workers)
    # Return source's dataset DataLoader object
    return loader, source_dataset


def get_target_domain(target_name, args):
    # Define root folder to store target dataset
    root = CURRENT_DIR_PATH + "/target"
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    target_img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # Define image target domain transformation
    # Define target dataset
    if target_name == 'amazon' or target_name == 'dslr' or target_name == 'webcam':
        train_target_dataset = Office31(target_name,
                                        transform=target_img_transforms)
        test_target_dataset = Office31(target_name,
                                       transform=target_img_transforms)
        args.num_classes = 31

    elif target_name == 'Art' or target_name == 'Clipart' or target_name == 'Product' or target_name == 'Real World':
        train_target_dataset = OfficeHome(
            target_name, transform=target_img_transforms)
        test_target_dataset = OfficeHome(
            target_name, transform=target_img_transforms)
        args.num_classes = 65

    elif target_name == "real" or target_name == "synthetic":
        train_target_dataset = get_dataset_visda(args, 'train_target', None)
        test_target_dataset = get_dataset_visda(args, 'test_target', None)
        args.num_classes = 12

    elif target_name == "sketch" or target_name == "clipart":
        train_target_dataset = DomainNet(
            target_name, args, transform=target_img_transforms)
        test_target_dataset = DomainNet(
            target_name, args, transform=target_img_transforms, test=True)

    # Define target dataloader
    if target_name == "real" or target_name == "synthetic":
        train_idx, val_idx = train_test_split(list(range(len(
            train_target_dataset))), test_size=0.2, random_state=42, stratify=train_target_dataset.targets)
    elif target_name != "sketch" or target_name != "clipart":
        train_idx, val_idx = train_test_split(list(range(len(
            train_target_dataset))), test_size=0.2, random_state=42, stratify=train_target_dataset.label)
    if target_name != "sketch" and target_name != "clipart":
        train_dataset = Subset(train_target_dataset, train_idx)
        test_dataset = Subset(test_target_dataset, val_idx)
    else:
        train_dataset = train_target_dataset
        test_dataset = test_target_dataset
        print("Number of images in test dataset of domain ",
              target_name, ":", len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Return target's dataset DataLoader object
    return (train_dataset, train_loader), (test_dataset, test_loader)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# DomainNet
def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


def return_classlist(image_list):
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
    return label_list
