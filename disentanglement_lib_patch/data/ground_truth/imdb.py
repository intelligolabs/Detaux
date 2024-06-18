# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import platform
import pickle
import torch
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
from six.moves import range
import h5py
from sklearn.model_selection import train_test_split
import itertools
from pathlib import Path
from PIL import Image
from scipy.io import loadmat

DT_PATH = ""

class IMDB(ground_truth_data.GroundTruthData):
    """
    IMDB dataset.
    """
    factor_sizes = [100, 2]
    def __init__(self, split_type, use_all=False):
        self.dataset_path = DT_PATH
        metadata = loadmat(os.path.join(self.dataset_path, 'imdb.mat'))['imdb'][0][0]
        gender = metadata[3].flatten().astype(int)
        age = metadata[10].flatten().astype(int)
        cid = metadata[9].flatten().astype(int)
        second_face_score = metadata[7].flatten()
        img_paths = np.array([str(x[0]) for x in metadata[2].flatten()])

        # Remove samples with more than 1 face or ages outside the range 0-99
        correct_idx = np.where((np.isnan(second_face_score)) & ((age>=0) & (age<100)) & (~np.isnan(gender)) & (gender >= 0))[0]
        img_paths = img_paths[correct_idx]
        age = age[correct_idx]
        gender = gender[correct_idx]
        
        # Train-test split
        cid = cid[correct_idx]
        # unique_ids = np.unique(cid)
        # train_ids, test_ids = train_test_split(unique_ids, train_size=0.8, random_state=21)
        # train_idx, test_idx = np.where(np.isin(cid, train_ids))[0], np.where(np.isin(cid, test_ids))[0]

        if use_all:
            self.imgs = img_paths
            self.age = age
            self.gender = gender
            self.cid = cid
        else:
            partition_idx = np.load(os.path.join(self.dataset_path, f"{split_type}_idx.npy"))
            self.imgs = img_paths[partition_idx]
            self.age = age[partition_idx]
            self.gender = gender[partition_idx]
            self.cid = cid[partition_idx]

            # if split_type == "train":
            #     self.imgs = img_paths[train_idx]
            #     self.age = age[train_idx]
            #     self.gender = gender[train_idx]
            #     self.cid = cid[train_idx]
            # else:
            #     self.imgs = img_paths[test_idx]
            #     self.age = age[test_idx]
            #     self.gender = gender[test_idx]
            #     self.cid = cid[test_idx]

        self.y = np.stack((self.age, self.gender), axis=-1)
      
        print("Loaded data")

        # Prepare Disentanglement lib info
        self.num_samples = self.imgs.shape[0]
        self.latent_factor_indices = np.arange(len(self.factor_sizes))
        self.num_total_factors = len(self.factor_sizes)
        self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                        self.latent_factor_indices)
        # self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
        #     self.factor_sizes)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return [256, 256, 3]

    def get_len(self):
        return self.num_samples
    
    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def get_image(self, item):
        return Image.open(os.path.join(self.dataset_path, self.imgs[item])).convert('RGB').resize((256,256)), self.y[item]

    def get_second_image(self, first_observation_factors, random_state):
        # Random state chooses if the factor should be the some or not
        factor_of_variation = random_state.randint(self.num_total_factors)
        equal_factors = np.delete(self.latent_factor_indices, factor_of_variation)
        sample_space = np.where(np.array(self.y[:, factor_of_variation] != first_observation_factors[factor_of_variation]) & \
            np.all(self.y[:, equal_factors] == first_observation_factors[equal_factors], axis=1))[0]
        
        second_observation_idx = None
        if len(sample_space) > 0:
            second_observation_idx = random_state.choice(sample_space)
        else:
            backup_sample_space = np.where(np.array(self.y[:, factor_of_variation] != first_observation_factors[factor_of_variation]) & \
                np.all(self.y[:, equal_factors] >= first_observation_factors[equal_factors]-10, axis=1))[0]
            second_observation_idx = random_state.choice(backup_sample_space)

        return Image.open(os.path.join(self.dataset_path, self.imgs[second_observation_idx])).convert('RGB').resize((256,256)), self.y[second_observation_idx]


class IMDB_mtl(IMDB):
    def __init__(self, split_type):
        super(IMDB_mtl, self).__init__(split_type)


class IMDB_all(IMDB):
    def __init__(self, split_type):
        super(IMDB_all, self).__init__(split_type, True)