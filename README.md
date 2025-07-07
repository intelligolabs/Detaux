# Disentangled Latent Spaces Facilitate Data-Driven Auxiliary Learning #

Official implementation of the paper [Disentangled Latent Spaces Facilitate Data-Driven Auxiliary Learning](https://intelligolabs.github.io/Detaux/) accepted at the 61st Design Automation Conference (DAC 2024).

## Installation ##
**1. Repository setup:**
* `$ git clone https://github.com/intelligolabs/Detaux`
* `$ cd Detaux`
* `$ git clone https://github.com/google-research/disentanglement_lib.git`
* `$ mv disentanglement_lib disentanglement_library`
* `$ cp -r disentanglement_lib_patch/* disentanglement_library/disentanglement_lib/`
* Download 3dshapes.h5 from https://console.cloud.google.com/storage/browser/3d-shapes;tab=objects?prefix=&forceOnObjectsSortingFiltering=false

**2. Conda enviroment setup:**
* `$ conda create -n detaux python=3.7`
* `$ conda activate detaux`
* `$ python -m pip install pytorch-lightning==1.9.4`
* `$ cd disentanglement_library/`
* `$ python -m pip install -v -e .`
* `$ python -m pip install tensorflow-gpu==1.14`
* `$ python -m pip install --upgrade tensorboard`
* `$ cd ../`
* `$ python -m pip install wandb`
* `$ pip install torchvision`

## Run Detaux ##
1. To run the disentanglement part, use the file `detaux.py`. In particular, `launch_dis.sh` it contains one example of a launch script that you can use to modify the default configuration directly.
2. To run the clustering part, use the file `clustering.py`.
3. Finally, with the file `aux_learning.py`, you will be able to perform the auxiliary learning phase with the new labels discovered in step 2.

## Authors ##
Geri Skenderi<sup>1</sup>, Luigi Capogrosso<sup>2</sup>, Andrea Toaiari<sup>2</sup>, Matteo Denitto<sup>3</sup>, Franco Fummi<sup>2</sup>, Simone Melzi<sup>4</sup>

<sup>1</sup> *Bocconi University, Bocconi Institute for Data Science and Analytics, Milan, Italy*

<sup>2</sup> *University of Verona, Dept. of Engineering for Innovation Medicine, Verona, Italy*

<sup>3</sup> *HUMATICS - SYS-DAT Group, Verona, Italy*

<sup>4</sup> *University of Milano-Bicocca, Dept. of Informatics, Systems and Communication, Milan, Italy*

<sup>1</sup> `geri.skenderi@unibocconi.it`, <sup>2</sup> `name.surname@univr.it`, <sup>3</sup> `matteo.denitto@sys-datgroup.com`, <sup>4</sup> `simone.melzi@unimib.it`,

## Citation ##
If you use [**Detaux**](https://arxiv.org/abs/2310.09278), please, cite the following paper:
```
@Article{skenderi2023disentangled,
  title   = {{Disentangled Latent Spaces Facilitate Data-Driven Auxiliary Learning}},
  author  = {Skenderi, Geri and Capogrosso, Luigi and Toaiari, Andrea and Denitto, Matteo and Fummi, Franco and Melzi, Simone and Cristani, Marco},
  journal = {arXiv preprint arXiv:2310.09278},
  year    = {2023}
}
```

## Credits ##
We want to thank Marco Fumero for the repository [PMPdisentanglement](https://github.com/marc0git/PMPdisentanglement), which provides us with the scripts used to manage the disentanglement part.
