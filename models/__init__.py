#!/usr/bin/env python
# -*- coding: utf-8 -*-

from models.simpleconv_ae import SimpleConv64, SimpleConv64D
from models.resnet_ae import ResNetEncoder, ResNetDecoder


conv_models = {
    "simpleconv": [SimpleConv64, SimpleConv64D],
    "resnetae": [ResNetEncoder, ResNetDecoder],
}
