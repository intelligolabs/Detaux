#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch

import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import Dataset

import disentanglement_lib.data.ground_truth.named_data as named_data


# Implemented datasets:
# -- 3D Shapes. dataset_names = shapes3d_custom_detangle; idx = 4.
dataset_names = ['shapes3d', 'shapes3d_less', 'shapes3d_random',
                 'shapes3d_custom_mtl', 'shapes3d_custom_detangle', 'jaffe',
                 'cifar10', 'cifar10SR', 'woof_mtl', 'woof_all', 'imdb_mtl',
                 'imdb_all', 'faces', 'cub', 'medic', 'cars', 'pets', 'cifar',
                 "svhn"]


def simple_dynamics(z, ground_truth_data, random_state,
                    return_index=False, k=1):
    """Create the pairs."""
    if k == -1:
        k_observed = random_state.randint(1, ground_truth_data.num_factors)
    else:
        k_observed = k
    # index_list = random_state.choice(
    #     z.shape[1], random_state.choice([1, k_observed]), replace=False)
    index_list = random_state.choice(z.shape[1], k_observed, replace=False)

    idx = -1
    for index in index_list:
        z[:, index] = random_state.choice([value for value in range(ground_truth_data.factors_num_values[index]) if value != int(z[:, index])])
        idx = index
    if return_index:
        return z, idx
    return z, k_observed


# Class for synthetic disentanglement datasets implemeted via disentanglement lib with known factors of variation.
class BasicDatasetVariableK(Dataset):
    def __init__(self, dataset_name="shapes3d", seed=0, factors=None, fraction=1, k=-1, split_type="train", transform=transforms.Compose([transforms.ToTensor()]), tasks=[], full_random=False):
        self.dataset_name = dataset_name
        self.dataset = named_data.get_named_ground_truth_data(self.dataset_name, split_type, tasks)
        if factors is None:
            factors = self.dataset.latent_factor_indices
        self.factors = factors
        self.fraction = fraction
        self.k=k
        self.randomstate = np.random.RandomState(seed)
        self.split_type = split_type
        self.transform = transform
        self.tasks = tasks
        self.full_random = full_random

    def __len__(self) -> int:
        return int(self.dataset.get_len() * self.fraction)

    def __getitem__(self, item: int):
        ground_truth_data = self.dataset
        sampled_observation, sampled_factors = ground_truth_data.get_image(item)
        sampled_factors = np.array(sampled_factors).reshape((1, -1))
        next_factors, index = simple_dynamics(sampled_factors.copy(),
                                                ground_truth_data,
                                                self.randomstate,
                                                return_index=True,
                                                k=self.k)
        next_observation = ground_truth_data.sample_observations_from_factors(next_factors, self.randomstate)
        sampled_observation = self.transform(sampled_observation).permute(1, 2, 0)
        next_observation = self.transform(next_observation[0]).permute(1, 2, 0)

        return sampled_observation, next_observation, torch.as_tensor(sampled_factors!=next_factors).int(), \
            torch.as_tensor(sampled_factors), torch.as_tensor(next_factors)


# Class for more broad dataset support which manually does the sampling of the second factors of variation.
class DatasetVariableK(Dataset):
    def __init__(self, dataset_name="shapes3d", seed=0, factors=None, fraction=1, k=-1, split_type="train", transform=transforms.Compose([transforms.ToTensor()]), tasks=[], full_random=False):
        self.dataset_name = dataset_name
        self.dataset = named_data.get_named_ground_truth_data(self.dataset_name, split_type, tasks)
        if factors is None:
            factors = self.dataset.latent_factor_indices
        self.factors = factors
        self.fraction = fraction
        self.k=k
        self.randomstate = np.random.RandomState(seed)
        self.split_type = split_type
        self.transform = transform
        self.tasks = tasks
        self.full_random = full_random

    def __len__(self) -> int:
        return int(self.dataset.get_len() * self.fraction)

    def __getitem__(self, item: int):
        ground_truth_data = self.dataset

        sampled_observation, sampled_factors = ground_truth_data.get_image(item)
        if 'woof' in self.dataset_name or 'cifar' in self.dataset_name:
            next_observation, next_factors = ground_truth_data.get_second_image(sampled_factors, self.randomstate, 0.5)
        else:
            next_observation, next_factors = ground_truth_data.get_second_image(sampled_factors, self.randomstate, self.full_random)

        sampled_observation = self.transform(sampled_observation).permute(1, 2, 0)
        next_observation = self.transform(next_observation).permute(1, 2, 0)

        return sampled_observation, next_observation, torch.as_tensor(sampled_factors!=next_factors).int(), \
            torch.as_tensor(sampled_factors), torch.as_tensor(next_factors)


# Used for the limited dataset, where we can control the observation factor and the probability of the oracle.
class LimitedDatasetVariableK(Dataset):
    def __init__(self, dataset_name="shapes3d", seed=0, factors=None, fraction=1, k=-1, split_type="train", transform=transforms.Compose([transforms.ToTensor()]), observation_factor=2, oracle_probability=0.5, random_item=False):
        self.dataset_name = dataset_name
        self.dataset = named_data.get_named_ground_truth_data(self.dataset_name, split_type)
        if factors is None:
            factors = self.dataset.latent_factor_indices
        self.factors = factors
        self.fraction = fraction
        self.k=k
        self.randomstate = np.random.RandomState(seed)
        self.split_type = split_type
        self.transform = transform
        self.observation_factor = observation_factor
        assert self.observation_factor in self.factors
        self.oracle_probability = oracle_probability
        self.random_item = random_item

    def __len__(self) -> int:
        return self.dataset.get_len()

    def __getitem__(self, item: int):
        # observation_factor: is the only factor over which we have control.
        # oracle_probability: indicates the probability that the loader returns a pair in which the observation_factor is different.
        # random_item: is true if you want the loader to extract the first image of each pair at random.
        #              If false, it is guaranteed that all the images in the dataset will be seen at least once.
        ground_truth_data = self.dataset
        if self.random_item:
            sampled_factors = self.randomstate.randint(ground_truth_data.factor_sizes)
            sampled_observation = ground_truth_data.sample_observations_from_factors(sampled_factors, self.randomstate)[0]
        else:
            sampled_observation, sampled_factors = ground_truth_data.get_image(item)

        if 'woof' or 'cifar' or 'faces' or 'cub' or 'medic' or 'cars' or 'pets' in self.dataset_name:
            next_observation, next_factors = ground_truth_data.get_second_image(sampled_factors, self.randomstate, self.oracle_probability)
        else:
            next_factors = self.randomstate.randint(ground_truth_data.factor_sizes)
            equal_observation_factor_probability = self.randomstate.choice([1, 2], p=[1-self.oracle_probability, self.oracle_probability])
            if equal_observation_factor_probability == 1:
                next_factors[self.observation_factor] = sampled_factors[self.observation_factor]
            else:
                next_factors[self.observation_factor] = self.randomstate.choice([value for value in range(ground_truth_data.factor_sizes[self.observation_factor]) if value != sampled_factors[self.observation_factor]])
            next_observation = ground_truth_data.sample_observations_from_factors(next_factors, self.randomstate)[0]

        sampled_observation = self.transform(sampled_observation).permute(1, 2, 0)
        next_observation = self.transform(next_observation).permute(1, 2, 0)


        return sampled_observation, next_observation, torch.as_tensor(sampled_factors!=next_factors).int(), \
            torch.as_tensor(sampled_factors), torch.as_tensor(next_factors)
