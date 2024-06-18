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
import torch
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
from six.moves import range
import torchvision

class CIFAR(ground_truth_data.GroundTruthData):
    """
    CIFAR10 dataset.
    """

    def __init__(self, split_type, equal_factors=False):
        """ load all of cifar """
        xs = []
        ys = []

        train_flag = True if split_type == 'train' else False
        
        # Load CIFAR-10 dataset
        dataset = torchvision.datasets.CIFAR10(root='/media/data/gskenderi/', train=train_flag, download=True)

        for pil_img, labels in dataset:
            xs.append(pil_img)
            ys.append(labels)
        
        ys = np.array(ys)
        self.X, self.y = xs, ys
        print("Loaded data")

        # Prepare Disentanglement lib info
        self.tasks = None
        self.num_samples = len(self.y)
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
    
    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return self.X[indices]

    def get_len(self):
        return self.num_samples

    def get_image(self, item):
        return self.X[item], self.y[item]

    def get_second_image(self, first_observation_factors, random_state, oracle_probability):
        #  Random state chooses if the factor should be the some or not
        same_factor = random_state.rand(1)
        if same_factor >= oracle_probability:
            sample_space = np.where(self.y == first_observation_factors)[0]
        else:
            sample_space = np.where(self.y != first_observation_factors)[0]
        
        second_observation_idx = random_state.choice(sample_space)

        return self.X[second_observation_idx], self.y[second_observation_idx]