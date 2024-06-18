#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import torch
import itertools
import torchvision

import numpy as np
import pandas as pd

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from tools.custom_transformations import SaltAndPepperNoise


DT_PATH = "/media/data/atoaiari/datasets/imagewoof2-320"
LBL_PATH = "disentanglement_lib_patch/data/ground_truth/imagenet_labels.txt"


def load_class(classpath, dt_path, label_dict, transform):
    # Load a single class of the dataset based on the file structure.
    lbl = label_dict[classpath]
    X, Y = [], []
    full_classpath = os.path.join(dt_path, classpath)
    for img in os.listdir(full_classpath):
        img_path = os.path.join(full_classpath, img)
        img = transform(Image.open(img_path).convert('RGB'))
        X.append(img.unsqueeze(0))
        Y.append(lbl)

    return torch.vstack(X), Y


class Aux3DShapes(Dataset):
    """Shapes3D dataset.

    The data set was originally introduced in "Disentangling by Factorising".

    The ground-truth factors of variation are:
    0 - floor color (10 different values)
    1 - wall color (10 different values)
    2 - object color (10 different values)
    3 - object size (8 different values)
    4 - object type (4 different values)
    5 - azimuth (15 different values)
    """

    def __init__(
        self,
        mined_labels,
        split_type,
        noise=False,
        main_task_idx=2,
        split_factor_idx=[0, 1],
        num_test_set_exclusive = [2, 2],
        factor_sizes=[10, 10, 10, 8, 4, 15],
        dataset_path='data/3dshapes.h5',
        reduced=True
    ):
        with h5py.File(dataset_path, 'r') as dataset:
            images = dataset['images'][()]
            labels = dataset['labels'][()]
        n_samples = images.shape[0]
        images = images.reshape([n_samples, 64, 64, 3]).astype(np.float32) / 255.
        labels = labels.reshape([n_samples, 6])

        integer_labels = np.array([list(i) for i in itertools.product(*[range(x) for x in factor_sizes])])

        # Reduce dataset size (optional).
        if reduced:
            factor_sizes = np.array([f//2 for f in factor_sizes])
            factor_sizes = factor_sizes.reshape(1, -1)
            reduced_idx = np.where(np.all(integer_labels < factor_sizes, axis=1))[0]
            images = images[reduced_idx]
            integer_labels = integer_labels[reduced_idx]

        if noise:
            self.noise = SaltAndPepperNoise(noiseType="SnPn", treshold=0.08)

        # Load mined labels.
        mined_labels = np.load(mined_labels)
        main_task_labels = integer_labels[:, main_task_idx]

        # AUX metadata.
        self.X = images
        self.y = np.stack([main_task_labels, mined_labels]).T
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [len(np.unique(self.y[:, task])) for task in self.task_ids]

        # Split dataset into train and test.
        subset_factor_sizes = np.array(factor_sizes).flatten()
        if split_factor_idx and num_test_set_exclusive:
            print(f"Dataset splitted on factors {split_factor_idx} by {num_test_set_exclusive}")
            assert not np.in1d(main_task_idx, split_factor_idx).any() and len(split_factor_idx) == len(num_test_set_exclusive)
            subset_factor_sizes[split_factor_idx] -= num_test_set_exclusive

        subset_idxs = None
        if split_type == 'train':
            subset_idxs = np.where(
                (integer_labels[:, split_factor_idx] < subset_factor_sizes[split_factor_idx]).sum(1) > 0
                )[0]
        else:
            subset_idxs = np.where(
                (integer_labels[:, split_factor_idx] < subset_factor_sizes[split_factor_idx]).sum(1) == 0
                )[0]

        self.X = self.X[subset_idxs]
        self.y = self.y[subset_idxs]


    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        img = torch.tensor(self.X[index])
        labels = torch.tensor(self.y[index])

        if self.noise:
            img = self.noise(img)

        return img.permute(2,0,1), labels


class AuxCub(Dataset):
    def __init__(
        self,
        data_path,
        mined_labels,
        split_type,
        transforms,
        main_task_idx=0,
    ):
        super().__init__()
        self.dataset_path = data_path
        self.tv_transforms = transforms
        self.split_type = split_type

        images = pd.read_csv(os.path.join(self.dataset_path, 'images.txt'),
                             sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.split_type == 'train':
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        y = np.stack([self.data['target']], axis=1).flatten() - 1

        # Load mined labels.
        mined_labels = np.load(mined_labels)
        if split_type == 'train':
            mined_labels = mined_labels[:len(y)]
        else:
            mined_labels = mined_labels[-len(y):]

        # AUX metadata.
        self.y = np.stack([y, mined_labels], axis=1)
        self.imgs = self.data.loc[:, 'filepath'].values

        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [len(np.unique(self.y[:, task])) for task in self.task_ids]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        img = Image.open(self.dataset_path / 'images' / self.imgs[index]).convert('RGB')

        return self.tv_transforms(img), self.y[index]


class AuxFaces(Dataset):
    def __init__(
        self,
        data_path,
        mined_labels,
        split_type,
        transforms,
        main_task_idx=2,
    ):
        super().__init__()
        self.dataset_path = data_path
        self.tv_transforms = transforms
        self.split_type = split_type

        images = []
        ages = []
        genders = []
        expressions = []

        for img in sorted(os.listdir(os.path.join(data_path, split_type))):
            # Read person ID, age, and expression from filename.
            img_labels = img.split("_")
            ages.append(img_labels[1])
            genders.append(img_labels[2])
            expressions.append(img_labels[3])

            # Save the image.
            images.append(img)

        # Prepare dataset specific information.
        ages_lbls = {x:e for e, x in enumerate(sorted(set(ages)))}
        gender_lbls = {x:e for e, x in enumerate(sorted(set(genders)))}
        expression_lbls = {x:e for e, x in enumerate(sorted(set(expressions)))}

        age_lbl_encoded = [ages_lbls[x] for x in ages]
        gender_lbl_encoded = [gender_lbls[x] for x in genders]
        expression_lbl_encoded = [expression_lbls[x] for x in expressions]
        y = np.stack([age_lbl_encoded, gender_lbl_encoded, expression_lbl_encoded], axis=1)

        # Load mined labels.
        mined_labels = np.load(mined_labels)
        if split_type == 'train':
            mined_labels = mined_labels[:len(y)]
        else:
            mined_labels = mined_labels[-len(y):]

        # AUX metadata.
        self.y = np.stack([y[:, main_task_idx], mined_labels], axis=1)
        self.X = images

        # AUX metadata -> does it not proceed to subset?
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [len(np.unique(self.y[:, task])) for task in self.task_ids]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        img = Image.open(self.dataset_path / self.split_type / self.X[index]).convert('RGB')

        return self.tv_transforms(img), self.y[index]


class AuxWoof(Dataset):
    def __init__(
        self,
        data_path,
        mined_labels,
        split_type,
        transforms,
        main_task_idx=0,
    ):
        super().__init__()
        self.dataset_path = data_path
        self.tv_transforms = transforms
        self.split_type = split_type

        """ load list of labels """
        with open(LBL_PATH) as f:
            label_dict = {line.split(" ")[0]: int(line.split(" ")[1]) for line in f}

        xs = []
        ys = []

        print(f"Loading imagewoof {self.split_type} classes...")
        dt_path = [os.path.join(DT_PATH, split_type)]

        for path in dt_path:
            for fd in os.listdir(path):
                X, Y = load_class(fd, path, label_dict, self.tv_transforms)
                xs.append(X)
                ys.append(Y)

        Xdata = torch.vstack(xs)
        Ydata = np.concatenate(ys)
        unique_lbl_dict = {v: i for i, v in enumerate(np.unique(Ydata))}
        self.X, y = Xdata, np.array([unique_lbl_dict[x] for x in Ydata])

        # Load mined labels.
        mined_labels = np.load(mined_labels)
        if split_type == 'train':
            mined_labels = mined_labels[:len(y)]
        else:
            mined_labels = mined_labels[-len(y):]

        # AUX metadata.
        self.y = np.stack([y, mined_labels], axis=1)

        # AUX metadata -> does it not proceed to subset?
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [len(np.unique(self.y[:, task])) for task in self.task_ids]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class AuxMedic(Dataset):
    def __init__(
        self,
        data_path,
        mined_labels,
        split_type,
        transforms,
        main_task_idx=1,
    ):
        super().__init__()
        self.dataset_path = data_path
        self.tv_transforms = transforms
        self.split_type = split_type

        # Load metadata.
        if self.split_type == 'train':
            data = pd.read_csv(os.path.join(self.dataset_path, 'MEDIC_train.tsv'), sep='\t')
        else:
            data = pd.read_csv(os.path.join(self.dataset_path, 'MEDIC_test.tsv'), sep='\t')

        # Get info from pandas df.
        images = data.loc[:, 'image_path'].values
        y1 = data.loc[:, 'damage_severity'].values
        y2 = data.loc[:, 'disaster_types'].values

        # Filter data.
        correct_idx = np.where((y2 != 'not_disaster') & (y2 != 'other_disaster'))[0]
        images = images[correct_idx]
        y1 = y1[correct_idx]
        y2 = y2[correct_idx]

        # Prepare dataset specific information.
        y1_lbls = {x:e for e, x in enumerate(sorted(set(y1)))}
        y2_lbls = {x:e for e, x in enumerate(sorted(set(y2)))}
        y1_lbl_encoded = [y1_lbls[x] for x in y1]
        y2_lbl_encoded = [y2_lbls[x] for x in y2]
        y = np.stack([y1_lbl_encoded, y2_lbl_encoded], axis=1)
        self.imgs = images

        # Load mined labels.
        mined_labels = np.load(mined_labels)
        if split_type == 'train':
            mined_labels = mined_labels[:len(y)]
        else:
            mined_labels = mined_labels[-len(y):]

        self.y = np.stack([y[:, main_task_idx], mined_labels], axis=1)

        # AUX metadata -> does it not proceed to subset?
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [len(np.unique(self.y[:, task])) for task in self.task_ids]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        imgs = Image.open(self.dataset_path / self.imgs[index]).convert('RGB')

        return self.tv_transforms(imgs), self.y[index]


class AuxCars(Dataset):
    def __init__(
        self,
        mined_labels,
        split_type,
        transforms,
        data_path="/media/data/lcapogrosso/CARS",
        main_task_idx=0,
    ):
        super().__init__()
        self.dataset_path = data_path
        self.partition = split_type
        self.tv_transforms = transforms

        # Load image folder paths and annotations.
        self.image_folder = os.path.join(data_path, f'cars_{split_type}')
        annotations = loadmat(os.path.join(data_path, 'devkit', f'cars_{split_type}_annos_withlabels.mat'))
        annotations = annotations['annotations'][0]

        y = [annot[-2].item() for annot in annotations]     # Build class label list.
        img = [annot[-1].item() for annot in annotations]   # Build img files list.
        y = np.array(y)
        y = y - 1

        # Load mined labels.
        mined_labels = np.load(mined_labels)
        if split_type == 'train':
            mined_labels = mined_labels[:len(y)]
        else:
            mined_labels = mined_labels[-len(y):]

        # AUX metadata.
        self.img = img
        self.y = np.stack([y, mined_labels], axis=1)

        # AUX metadata -> does it not proceed to subset?
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [len(np.unique(self.y[:, task])) for task in self.task_ids]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        path = os.path.join(self.image_folder, self.img[index])
        img = Image.open(path).convert('RGB')

        return self.tv_transforms(img), self.y[index]


class AuxPets(Dataset):
    def __init__(
        self,
        mined_labels,
        split_type,
        transforms,
        data_path="/media/data/gskenderi/oxford-iiit-pet",
        main_task_idx=0,
    ):
        super().__init__()
        self.dataset_path = data_path
        self.partition = split_type
        self.tv_transforms = transforms

        # Load image folder paths and annotations.
        partition_str = 'trainval' if split_type=='train' else 'val'
        self.image_folder = os.path.join(data_path, 'images')
        annot_list = os.path.join(data_path, 'annotations', f'{partition_str}.txt')
        annotations = pd.read_csv(annot_list, header=None, sep=' ')
        img = annotations.iloc[:, 0].values # Build img files array
        y = annotations.iloc[:, 1].values   # Build class label array
        y = y - 1

        # Load mined labels.
        mined_labels = np.load(mined_labels)
        if split_type == 'train':
            mined_labels = mined_labels[:len(y)]
        else:
            mined_labels = mined_labels[-len(y):]

        # AUX metadata.
        self.img = img
        self.y = np.stack([y, mined_labels], axis=1)

        # AUX metadata -> does it not proceed to subset?
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [len(np.unique(self.y[:, task])) for task in self.task_ids]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        sample = self.img[index]
        path = os.path.join(self.image_folder, f'{sample}.jpg')
        img = Image.open(path).convert('RGB')

        return self.tv_transforms(img), self.y[index]


class AuxCifar(Dataset):
    def __init__(
        self,
        mined_labels,
        split_type,
        transforms,
        data_path="/media/data/gskenderi/"
    ):
        super().__init__()
        self.partition = split_type
        self.tv_transforms = transforms

        xs, ys = [], []
        train_flag = True if split_type == 'train' else False

        # Load CIFAR-10 dataset.
        dataset = torchvision.datasets.CIFAR10(root=data_path, train=train_flag, download=True)

        for pil_img, labels in dataset:
            xs.append(pil_img)
            ys.append(labels)

        ys = np.array(ys)
        print("Loaded data")

        # Load mined labels.
        mined_labels = np.load(mined_labels)
        if split_type == 'train':
            mined_labels = mined_labels[:len(ys)]
        else:
            mined_labels = mined_labels[-len(ys):]

        # AUX metadata.
        self.img = xs
        self.y = np.stack([ys, mined_labels], axis=1)

        # AUX metadata -> does it not proceed to subset?
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [len(np.unique(self.y[:, task])) for task in self.task_ids]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        img = self.img[index]

        return self.tv_transforms(img), self.y[index]


class AuxSvhn(Dataset):
    def __init__(
        self,
        mined_labels,
        split_type,
        transforms,
        data_path="/media/data/gskenderi/"
    ):
        super().__init__()
        self.partition = split_type
        self.tv_transforms = transforms

        xs, ys = [], []

        # Load SVHN dataset.
        dataset = torchvision.datasets.SVHN(root=data_path, split=split_type, download=True)

        for pil_img, labels in dataset:
            xs.append(pil_img)
            ys.append(labels)

        ys = np.array(ys)
        print("Loaded data")

        # Load mined labels.
        mined_labels = np.load(mined_labels)
        if split_type == 'train':
            mined_labels = mined_labels[:len(ys)]
        else:
            mined_labels = mined_labels[-len(ys):]

        # AUX metadata.
        self.img = xs
        self.y = np.stack([ys, mined_labels], axis=1)

        # AUX metadata -> does it not proceed to subset?
        self.num_tasks = self.y.shape[1]
        self.task_ids = [i for i in range(self.num_tasks)]
        self.task_lbl_sizes = [len(np.unique(self.y[:, task])) for task in self.task_ids]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        img = self.img[index]

        return self.tv_transforms(img), self.y[index]
