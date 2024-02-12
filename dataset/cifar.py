import os
import logging
import math

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import datasets
from torchvision import transforms
import torch.utils.data as data
import pandas as pd
import random
from .utils import add_g, flip_image
import glob
from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)


def get_RAFDB(args, root):
    transform_labeled = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)) ])
    
    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    base_dataset = RafDataset(args, phase='train', transform=transform_labeled)
    
    train_labeled_idxs = np.array(range(0, 12271))
    
    train_labeled_dataset = RafDataset(args, phase='train', transform=transform_labeled)
    
    test_dataset = RafDataset(args, phase='test', transform=transform_val)
    
    return train_labeled_dataset, test_dataset

def get_lfw(args, root):
    transform_labeled = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)) ])
    
    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    base_dataset = lfwDataset(args, transform=transform_val)

    train_unlabeled_dataset = lfwDataset(args, transform=TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    # test_dataset = lfwDataset(args, transform=TransformFixMatch(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))


    return train_unlabeled_dataset


def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel', args.label_path), sep=' ', header=None)
        
        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('train')]
        else:
            df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
            dataset = df[df[name_c].str.startswith('test')]
            
        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        self.aug_func = [flip_image, add_g]
        self.file_paths = []
        self.clean = (args.label_path == 'list_patition_label.txt')
        
        for f in images_names:
            f = f.split(".")[0]
            f += '_aligned.jpg'
            file_name = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(file_name)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])
            
        image = image[:, :, ::-1]
        
        
        if not self.clean:    
            image1 = image
            image1 = self.aug_func[0](image)
            image1 = self.transform(image1)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)
        
        if self.clean:
            image1 = transforms.RandomHorizontalFlip(p=1)(image)

        return image, label

from PIL import Image

class lfwDataset(data.Dataset):
    def __init__(self, args, transform=None):
        """
        Args:
            root_dir (string): 데이터셋이 들어 있는 디렉토리 경로
            transform (callable, optional): 샘플에 적용될 transform
        """
        self.img_path = args.lfw_path
        self.transform = transform
        self.images = os.listdir(self.img_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_path, self.images[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image

DATASET_GETTERS = {'RAFDB': get_RAFDB,
                   'lfw': get_lfw}
