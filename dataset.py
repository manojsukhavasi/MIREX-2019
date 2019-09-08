import os
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

from albumentations import Compose, ShiftScaleRotate, GridDistortion
from albumentations.pytorch import ToTensor
from random_erasing import RandomErasing

random_erasing = RandomErasing()

albumentations_transform = Compose([
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.5),
    GridDistortion(),
    ToTensor()
])

def get_train_val_data(inp_txt):

    print('Reading training list file...')
    with open(inp_txt, 'r') as f:
        lines = f.readlines()

    lines = [line.rstrip() for line in lines]
    fnames = [line.split('\t')[0] for line in lines]
    labels = [line.split('\t')[-1] for line in lines]
    ref_labels = list(sorted(set(labels)))
    ref_labels_dict = {rl:i for i,rl in enumerate(ref_labels)}
    labels = [ref_labels_dict[l] for l in labels]
    
    return train_test_split(fnames, labels, shuffle=True, stratify=labels, test_size=0.20)

def get_test_data(inp_txt):

    with open(inp_txt, 'r') as f:
        lines = f.readlines()

    fnames = [line.rstrip() for line in lines]    
    return fnames


class AudioDataset(Dataset):

    def __init__(self, fnames, labels, root_dir, train=True):        
        self.fnames = fnames
        self.labels = labels
        self.melspec_dir = root_dir

        self.fnames = [
            os.path.splitext(os.path.basename(fname))[0]
            for fname in self.fnames]
        self.fnames = [
            self.melspec_dir + '/' + fname + '.npy'
            for fname in self.fnames]

        self.transform=None
        self.pil = transforms.ToPILImage()
        if train:
            self.transform = albumentations_transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):

        fname = self.fnames[idx]
        sample = np.load(fname)

        if self.transform:
            # min-max transformation
            this_min = sample.min()
            this_max = sample.max()
            sample = (sample - this_min) / (this_max - this_min)

            # randomly cycle the file
            i = np.random.randint(sample.shape[1])
            sample = np.concatenate((sample[:, i:], sample[:, :i]), axis=1)
            sample = torch.FloatTensor(np.expand_dims(sample, axis=0))

            # apply albumentations transforms
            sample = np.array(self.pil(sample))
            sample = self.transform(image=sample)
            sample = sample['image']
            sample = sample[None, :, :].permute(0, 2, 1)

            # apply random erasing
            sample = random_erasing(sample.clone().detach())

            # revert min-max transformation
            sample = (sample * (this_max - this_min)) + this_min

        else:
            sample = torch.FloatTensor(np.expand_dims(sample, axis=0))

        return sample, self.labels[idx]


class TestDataset(Dataset):
    def __init__(self, fnames, root_dir):
        self.fnames = fnames
        self.melspec_dir = root_dir

        self.fnames = [
            os.path.splitext(os.path.basename(fname))[0]
            for fname in self.fnames]
        self.fnames = [
            self.melspec_dir + '/' + fname + '.npy'
            for fname in self.fnames]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):

        fname = self.fnames[idx]
        sample = np.load(fname)

        i = np.random.randint(sample.shape[1])
        sample = np.concatenate((sample[:, i:], sample[:, :i]), axis=1)
        sample = torch.FloatTensor(np.expand_dims(sample, axis=0))

        return sample
