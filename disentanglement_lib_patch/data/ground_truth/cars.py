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
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

dataset_path = Path("")

class Cars(ground_truth_data.GroundTruthData):
    def __init__(self, split_type, use_all=False):
        super().__init__()
        self.dataset_path = dataset_path
        self.partition = split_type

        # Load image folder paths and annotations
        self.image_folder = os.path.join(dataset_path, f'cars_{split_type}')
        annotations = loadmat(os.path.join(dataset_path, 'devkit', f'cars_{split_type}_annos_withlabels.mat'))
        annotations = annotations['annotations'][0]


        y = [annot[-2].item() for annot in annotations] # Build class label list
        img = [annot[-1].item() for annot in annotations] # Build img files list
        y = np.array(y)
        y = y - 1

        # Prepare Disentanglement lib info
        self.img = img
        self.y = y
        self.split_type = split_type
        self.factor_sizes = [len(np.unique(self.y))]
        self.num_samples = len(self.y)
        self.latent_factor_indices = np.arange(len(self.factor_sizes))
        self.num_total_factors = len(self.factor_sizes)
        self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                        self.latent_factor_indices)

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
        sample = self.img[item]
        path = os.path.join(self.image_folder, sample)
        img = Image.open(path).convert('RGB')

        return img, self.y[item]

    def get_second_image(self, first_observation_factors, random_state, oracle_probability):
        same_factor = random_state.rand(1)
        if same_factor >= oracle_probability:
            sample_space = np.where(self.y == first_observation_factors)[0]
        else:
            sample_space = np.where(self.y != first_observation_factors)[0]

        second_observation_idx = random_state.choice(sample_space)
        sample = self.img[second_observation_idx]
        path = os.path.join(self.image_folder, sample)
        img = Image.open(path).convert('RGB')

        return img, self.y[second_observation_idx]
