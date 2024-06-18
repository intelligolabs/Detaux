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
from PIL import Image

dataset_path = Path("")

class CUB(ground_truth_data.GroundTruthData):
    def __init__(self, split_type, use_all=False):
        super().__init__()
        self.dataset_path = dataset_path
        self.partition = split_type

        images = pd.read_csv(os.path.join(self.dataset_path, 'images.txt'),
                             sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        
        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        self.data.target -= 1

        if self.partition == 'train':
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # Prepare Disentanglement lib info
        self.y = np.stack([self.data['target']], axis=1)
        self.split_type = split_type
        self.factor_sizes = [200]
        self.num_samples = len(self.data)
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
        sample = self.data.iloc[item]
        path = os.path.join(self.dataset_path, 'images/', sample.filepath)
        img = Image.open(path).convert('RGB')

        return img, self.y[item]

    def get_second_image(self, first_observation_factors, random_state, oracle_probability):
        same_factor = random_state.rand(1)
        if same_factor >= oracle_probability:
            sample_space = np.where(self.y == first_observation_factors)[0]
        else:
            sample_space = np.where(self.y != first_observation_factors)[0]
        
        second_observation_idx = random_state.choice(sample_space)
        sample = self.data.iloc[second_observation_idx]
        path = os.path.join(self.dataset_path, 'images/', sample.filepath)
        img = Image.open(path).convert('RGB')

        return img, self.y[second_observation_idx]
