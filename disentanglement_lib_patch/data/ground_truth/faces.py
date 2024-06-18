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

"""MEDIC dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import itertools
from pathlib import Path
from PIL import Image
import pandas as pd
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

dataset_path = Path("")

class FACES(ground_truth_data.GroundTruthData):
    def __init__(self, split_type, use_all=False):
        super().__init__()
        self.dataset_path = dataset_path

        images = []
        # ids = []
        ages = []
        genders = []
        expressions = []
        # picture_sets = []

        for img in sorted(os.listdir(os.path.join(dataset_path, split_type))):
            # Read person ID, age, and expression from filename.
            img_labes = img.split("_")
            # ids.append(img_labes[0])
            ages.append(img_labes[1])
            genders.append(img_labes[2])
            expressions.append(img_labes[3])
            # picture_sets.append(img_labes[4].split('.')[0])

            # Save the image.
            images.append(img)

        # Prepare dataset specific information.
        # id_lbls = {x:e for e, x in enumerate(sorted(set(ids)))}
        ages_lbls = {x:e for e, x in enumerate(sorted(set(ages)))}
        gender_lbls = {x:e for e, x in enumerate(sorted(set(genders)))}
        expression_lbls = {x:e for e, x in enumerate(sorted(set(expressions)))}
        # pset_lbls = {x:e for e, x in enumerate(sorted(set(picture_sets)))}

        # id_lbl_encoded = [id_lbls[x] for x in ids]
        age_lbl_encoded = [ages_lbls[x] for x in ages]
        gender_lbl_encoded = [gender_lbls[x] for x in genders]
        expression_lbl_encoded = [expression_lbls[x] for x in expressions]
        # pset_lbl_encoded = [pset_lbls[x] for x in picture_sets]
        # self.y = np.stack([id_lbl_encoded, gender_lbl_encoded, age_lbl_encoded, expression_lbl_encoded, pset_lbl_encoded], axis=1)
        self.y = np.stack([gender_lbl_encoded, age_lbl_encoded, expression_lbl_encoded], axis=1)
        self.imgs = images

        # Prepare Disentanglement lib info
        self.split_type = split_type
        self.factor_sizes = [2, 3, 6]
        self.num_samples = len(self.imgs)
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

    # def sample_observations_from_factors(self, factors, random_state):
    #     all_factors = self.state_space.sample_all_factors(factors, random_state)
    #     indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
    #     return self.images[indices][0]

    def get_image(self, item):
        return Image.open(dataset_path / self.split_type / self.imgs[item]).convert('RGB'), self.y[item]

    def get_second_image(self, first_observation_factors, random_state, oracle_probability=None):
        # Random state chooses if the factor should be the some or not
        factor_of_variation = random_state.randint(self.num_total_factors)
        equal_factors = np.delete(self.latent_factor_indices, factor_of_variation)
        sample_space = np.where(np.array(self.y[:, factor_of_variation] != first_observation_factors[factor_of_variation]) & \
            np.all(self.y[:, equal_factors] == first_observation_factors[equal_factors], axis=1))[0]
        second_observation_idx = random_state.choice(sample_space)

        return Image.open(dataset_path / self.split_type / self.imgs[second_observation_idx]).convert('RGB'), self.y[second_observation_idx]
