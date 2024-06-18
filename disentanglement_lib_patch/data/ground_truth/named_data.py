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

"""Provides named, gin configurable ground truth data sets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.data.ground_truth import cars3d
from disentanglement_lib.data.ground_truth import dsprites
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.data.ground_truth import mpi3d
from disentanglement_lib.data.ground_truth import norb
from disentanglement_lib.data.ground_truth import shapes3d
from disentanglement_lib.data.ground_truth import imagewoof
from disentanglement_lib.data.ground_truth import imdb
from disentanglement_lib.data.ground_truth import faces
from disentanglement_lib.data.ground_truth import cub
from disentanglement_lib.data.ground_truth import cars 
from disentanglement_lib.data.ground_truth import pets 
from disentanglement_lib.data.ground_truth import cifar10
from disentanglement_lib.data.ground_truth import svhn 
import gin.tf


@gin.configurable("dataset")
def get_named_ground_truth_data(name, split_type, tasks=[]):
    """Returns ground truth data set based on name.

    Args:
        name: String with the name of the dataset.

    Raises:
        ValueError: if an invalid data set name is provided.
    """

    if name == "dsprites_full":
        return dsprites.DSprites([1, 2, 3, 4, 5])
    elif name == "dsprites_noshape":
        return dsprites.DSprites([2, 3, 4, 5])
    elif name == "color_dsprites":
        return dsprites.ColorDSprites([1, 2, 3, 4, 5])
    elif name == "noisy_dsprites":
        return dsprites.NoisyDSprites([1, 2, 3, 4, 5])
    elif name == "scream_dsprites":
        return dsprites.ScreamDSprites([1, 2, 3, 4, 5])
    elif name == "smallnorb":
        return norb.SmallNORB()
    elif name == "cars3d":
        return cars3d.Cars3D()
    elif name == "mpi3d_toy":
        return mpi3d.MPI3D(mode="mpi3d_toy")
    elif name == "mpi3d_realistic":
        return mpi3d.MPI3D(mode="mpi3d_realistic")
    elif name == "mpi3d_real":
        return mpi3d.MPI3D(mode="mpi3d_real")
    elif name == "shapes3d":
        return shapes3d.Shapes3D()
    elif name == "shapes3d_less":
        return shapes3d.Shapes3D_less(factors=3)
    elif name == "shapes3d_random":
        return shapes3d.Shapes3D_random()
    elif name == "shapes3d_custom_mtl":
        return shapes3d.Shapes3D_custom_mtl(split_type, tasks)
    elif name == "shapes3d_custom_detangle":
        return shapes3d.Shapes3D_custom_detangle(split_type, tasks)
    elif name == "dummy_data":
        return dummy_data.DummyData()
    elif name == "woof_mtl":
        return imagewoof.Imagewoof_mtl(split_type)
    elif name == "woof_all":
        return imagewoof.Imagewoof_all(split_type)
    elif name == "imdb_mtl":
        return imdb.IMDB_mtl(split_type)
    elif name == "imdb_all":
        return imdb.IMDB_all(split_type)
    elif name == "faces":
        return faces.FACES(split_type)
    elif name == "cub":
        return cub.CUB(split_type)
    elif name == "cars":
        return cars.Cars(split_type)
    elif name == "pets":
        return pets.Pets(split_type)
    elif name == "cifar":
        return cifar10.CIFAR(split_type)
    elif name == "svhn":
        return svhn.SVHN(split_type)
    else:
        raise ValueError("Invalid data set name.")
