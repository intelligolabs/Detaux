# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

DT_PATH = ""
LBL_PATH = ""


def load_class(classpath, dt_path, label_dict):
    """ load a single class of the dataset based on the file structure """
    lbl = label_dict[classpath]
    X, Y = [], []
    full_classpath = os.path.join(dt_path, classpath)
    for img in os.listdir(full_classpath):
        img_path = os.path.join(full_classpath, img)
        pil_img = Image.open(img_path).convert('RGB').resize((320, 320))
        X.append(np.array(pil_img))
        Y.append(lbl)
    
    return X, Y


class Imagewoof(ground_truth_data.GroundTruthData):
    """
    Imagewoof dataset.
    """
    def __init__(self, split_type, use_all=False):
        """ load list of labels """
        with open(LBL_PATH) as f:
            label_dict = {line.split(" ")[0]: int(line.split(" ")[1]) for line in f}

        xs = []
        ys = []

        print("loading imagewoof classes...")
        dt_path = [os.path.join(DT_PATH, split_type)]
        if use_all:
            dt_path = [os.path.join(DT_PATH, splitt) for splitt in ['train', 'val']]
        
        for path in dt_path:
            for fd in os.listdir(path):
                X, Y = load_class(fd, path, label_dict)
                xs.append(X)
                ys.append(Y)

        Xdata = np.concatenate(xs).astype(np.float32) / 255.
        Ydata = np.concatenate(ys)
        unique_lbl_dict = {v: i for i, v in enumerate(np.unique(Ydata))}
        self.X, self.y = Xdata, np.array([unique_lbl_dict[x] for x in Ydata])
        del X, Y, xs, ys

        print("Loaded data")

        # Prepare Disentanglement lib info
        self.tasks = None
        self.num_samples = self.X.shape[0]
        self.factor_sizes = [10]
        self.latent_factor_indices = [0]
        self.num_total_factors = 1
        self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                        self.latent_factor_indices)
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
            self.factor_sizes)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return [32, 32, 3]

    def get_len(self):
        return self.num_samples
    
    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return self.X[indices]


    def get_image(self, item):
        return self.X[item], self.y[item]

    def get_second_image(self, first_observation_factors, random_state, oracle_probability):
        # Random state chooses if the factor should be the some or not
        same_factor = random_state.rand(1)
        if same_factor >= oracle_probability:
            sample_space = np.where(self.y == first_observation_factors)[0]
        else:
            sample_space = np.where(self.y != first_observation_factors)[0]
        
        second_observation_idx = random_state.choice(sample_space)

        return self.X[second_observation_idx], self.y[second_observation_idx]


class Imagewoof_mtl(Imagewoof):
    def __init__(self, split_type):
        super(Imagewoof_mtl, self).__init__(split_type)


class Imagewoof_all(Imagewoof):
    def __init__(self, split_type):
        super(Imagewoof_all, self).__init__(split_type, True)