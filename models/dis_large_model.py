#!/usr/bin/env python
# -*- coding: utf-8 -*-

import wandb
import torch
import random

import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
from torch import nn
from argparse import ArgumentParser
from distutils.util import strtobool
from torch.utils.data import DataLoader

from data.dis_datasets import *
from tools.custom_transformations import *


class LargeModel(pl.LightningModule):
    def __init__(self, hparams, encoder=None, encoders=None, decoder=None):
        # load_envs()
        super(LargeModel, self).__init__()
        self.save_hyperparameters(hparams)
        torch.set_default_tensor_type(torch.FloatTensor)

        self.lr= self.hparams.learning_rate
        self.latentdim = self.hparams.latentdim
        self.n_latent_factors = self.hparams.n_latent_factors

        self.projectors = encoders
        self.encoder = encoder
        self.decoder = decoder

        self.rec_criterion = nn.BCELoss(reduction="mean") if self.hparams.loss == 'bce' else nn.MSELoss(reduction="mean")

        self.aggregator = None
        self.eye_ld=None
        if self.projectors is None:
            self.projectors = nn.ModuleList()
            for _ in range(self.n_latent_factors):
                self.projectors.append(nn.Sequential(
                    nn.Conv2d(self.hparams.latentdim, self.hparams.latentdim, kernel_size=1),
                    nn.ReLU(True),
                    nn.Conv2d(self.hparams.latentdim, self.hparams.latentdim, kernel_size=1),
                    nn.ReLU(True),
                    nn.Conv2d(self.hparams.latentdim, self.hparams.latentdim, kernel_size=1)
                ))

        self.flattened_dim = self.hparams.latentdim*7*7

        self.save_hyperparameters(hparams)

    ######################## Helper functions ########################
    def pre_process(self,x):
        if len(x.shape)==3:
            x=x[None,...]
        x = x.permute(0,3,1,2)
        return x

    def post_process(self, x: torch.Tensor):
        x = self.squash(x.permute(0,2,3,1))
        return x

    def aggregate(self, x):
        if self.aggregator is not None:
            x = self.aggregator(torch.cat(x, -1))
            return x
        else:
            if type(x) is list or type(x) is tuple:
                x = torch.stack(x)
            return x.sum(0)

    def latent_encode(self, x):
        z_latents = []
        for i in range(self.n_latent_factors):
            tmp = self.projectors[i](x)
            z_latents.append(tmp)
        return z_latents

    def squash(self, x):
        return torch.sigmoid(x) if self.hparams.loss == 'bce' else x
    ################################################################

    def forward(self, xx: torch.Tensor):
        xx = self.pre_process(xx)                     # From [N, H, W, C] to [N, C, H, W].
        zz = self.encoder(xx)
        z_latents = self.latent_encode(zz)
        z_latents = [x.flatten(1) for x in z_latents] # Turn feature maps into flat vector representations.
        z_aggrs = self.aggregate(z_latents)
        outs = self.decoder(z_aggrs)
        outs = self.post_process(outs)                # From [N, C, H, W] to [N, H, W, C].

        return {
            "yy": outs,
            "z_latents": z_latents,
            "zz": zz,
            "z_aggrs": z_aggrs
        }


    def on_train_start(self) -> None:
        if self.hparams.wandb:
            self.logger.log_hyperparams(self.hparams)
        else:
            pass


    def training_step(self, batch, batch_idx):
        # Init losses.
        loss_distance = torch.tensor(0)
        loss_consistency = torch.tensor(0)
        loss_sparsity = torch.tensor(0)
        loss_reg = torch.tensor(0)

        x1, x2, _, gt1, gt2 = batch     # (x1, x2) are a batch_size pair of images.
        xx = torch.cat([x1,x2],0)       # We collapse the pairs in a single batch that is 2 x batch_size.
        outputs = self.forward(xx)

        yy, z_latents, zz, z_aggr = \
                    outputs["yy"],\
                    outputs["z_latents"], \
                    outputs["zz"], \
                    outputs["z_aggrs"]

        # yy are the reconstructed images.
        # z_latents are the projections s1...sn.
        # z_aggr is the aggregated subspace (the product space which is the goal).
        # z1 and z2 and zz contains z1^ and z2^.

        # Split in z1 and z2 in the first dimension.
        z_latents = torch.stack(z_latents,0)\
              .view(len(z_latents),2,-1,z_latents[0].shape[-1])\
              .transpose(0,1)
        zz = zz.view( *([2,-1] + list(zz.shape[1:])) )
        z_aggr = z_aggr.view( *([2,-1] + list(z_aggr.shape[1:])))

        epochs = self.hparams.max_epochs
        beta1 = min(np.exp((25/epochs)*(self.current_epoch - epochs*0.4)), 1)
        beta2 = min(np.exp((25/epochs)*(self.current_epoch - epochs*0.5)), 1)
        beta3 = 1-min(np.exp((25/epochs)*(self.current_epoch - epochs*0.4)), 1)

        # Reconstruction loss.
        loss_rec = self.rec_criterion(xx, yy)

        # Start only with reconstruction loss for the first 25% of epochs.
        if self.current_epoch>epochs*0.25:

            # Sparsity loss.
            if self.eye_ld is None:
                self.eye_ld = torch.eye(z_latents.shape[1]).to(z_latents.device)
            all_z = z_latents.permute(0,2,3,1)
            loss_sparsity = (all_z@(1-self.eye_ld)*all_z).abs().sum(-1).mean()

            # Consistency loss.
            if beta2>1e-2 and self.hparams.lambda_consistency>0:
                _, nl, bs, d = z_latents.shape

                z_misc = []
                z_latents_miscs_pre = []
                for i in range(self.n_latent_factors):
                    l = z_latents[1,:,:,:]+0
                    l[i] = z_latents[0,i,:,:]+0
                    z_misc.append(self.aggregate(l))
                    z_latents_miscs_pre.append(l)
                z_latents_miscs_pre = torch.stack(z_latents_miscs_pre,1).view(nl,nl*bs,d)

                # Subsample bs*2 combinations.
                idxs = torch.randperm(z_latents_miscs_pre.shape[1])[:bs*2]
                z_latents_miscs_pre =  z_latents_miscs_pre[:,idxs,:]
                z_miscs = torch.stack(z_misc,0).view(-1,d)[idxs] #nl,bs,d -> nlxbs,d
                decoded = self.decoder(z_miscs)
                z_latents_miscs = self.latent_encode(self.encoder(self.squash(decoded)))
                z_latents_miscs = [x.flatten(1) for x in z_latents_miscs]
                z_latents_miscs = torch.stack(z_latents_miscs,0)
                loss_consistency = F.mse_loss(z_latents_miscs, z_latents_miscs_pre, reduction="mean")

            # Distance loss.
            _, nl, bs, d = z_latents.shape
            dist = (z_latents[0]-z_latents[1]).pow(2).sum(-1).t()
            zs = z_latents.transpose(0,1).view(nl,2*bs,d)
            mean_dist = torch.cdist(zs,zs).mean([-2,-1])+1e-8
            dist_norm = dist/mean_dist

            mask = None
            if self.hparams.forced:
                forced_subspace = 0 # Decide where you wish to force the factors.
                # Compute the mask representing the oracle. The first subspace is exlcuded to accomodate the observation_factor.
                mask = torch.zeros_like(dist_norm).detach()
                mask[:, 1:] = torch.softmax(1e6*dist_norm[:, 1:], -1).detach()
                # Mask on the observation_factor. It is 1 if the observation_factor is different in the two images of each pair, 0 otherwise.
                if len(gt1.shape) > 1:
                    observation_mask = (gt1[:, self.hparams.observation] != gt2[:, self.hparams.observation]).detach()
                else:
                    observation_mask = (gt1 != gt2).detach()

                # If a pair of images differs in the observation_factor, the oracle prediction is replaced by [1, 0, ...].
                mask[observation_mask] = 0.
                mask[observation_mask, forced_subspace] = 1.
                mask = 1 - mask
            else:
                # Fully automated mask building procedure in case of no forced factors.
                mask = 1-torch.softmax(1e6*dist_norm,-1).detach()

            loss_distance = (dist*mask).mean() + 10*(0.05-dist*(1-mask)).relu().mean()

            # Regularization loss.
            if self.hparams.forced:
                # The mask is modified for consistency with the custom oracle.
                sel = torch.zeros_like(dist).detach()
                sel[:, 1:] = torch.softmax(1e6*dist_norm[:, 1:], -1).detach()
                sel[observation_mask] = 0.
                sel[observation_mask, forced_subspace] = 1.
                loss_reg = 1e2*(1/sel.shape[-1] - sel.mean(0)).pow(2).mean()
            else:
                sel = torch.softmax(1e6*dist_norm, -1).detach()  # Fix the regularization mask to be identical to the oracle used for the distance loss.
                loss_reg = 1e2*(1/sel.shape[-1] - sel.mean(0)).pow(2).mean()

        loss = loss_rec  \
               + beta1 * (self.hparams.lambda_distance * loss_distance +\
                          self.hparams.lambda_sparsity * loss_sparsity) \
               + beta2 *  self.hparams.lambda_consistency *  loss_consistency \
               + beta3 *  self.hparams.lambda_distribution * loss_reg

        self.log("train_loss", loss.item())
        self.log("loss_rec", loss_rec.item())
        self.log("loss_sparsity", loss_sparsity.item())
        self.log("loss_consistency", loss_consistency.item())
        self.log("loss_distance", loss_distance.item())
        self.log("loss_reg", loss_reg.item())
        # self.log("loss_kurt", loss_kurt.item())
        # self.log("avg_sq_kurtosis", (kurt**2).mean().item())
        self.log("beta1", beta1)
        self.log("beta2", beta2)
        self.log("beta3", beta3)

        if batch_idx == 0 and self.hparams.wandb:
            self.logger.experiment.log({"trainx1": [wandb.Image((xx[0, ...]).cpu().numpy(), caption=" trainx1"),
            wandb.Image((yy[0, ...]).detach().cpu().numpy(), caption="trainy1")]})

        return {"loss": loss}


    def training_epoch_end(self, outputs):
        epochs = self.hparams.max_epochs
        # if self.current_epoch>epochs*0.24:
        if self.current_epoch>epochs*0.01:
            idx1 = random.randint(0, len(self.train_set) - 1)
            idx2 = random.randint(0, len(self.train_set) - 1)

            # Select only the first images from the two pairs.
            x1 = self.train_set[idx1][0].cuda()
            x2 = self.train_set[idx2][0].cuda()

            topil = transforms.ToPILImage()
            I = self.get_dis_image(torch.stack([x1,x2])).cpu()
            if self.hparams.wandb:
                self.logger.experiment.log({'dis_training': [
                    wandb.Image(topil(I.permute(2,0,1)), caption="dis_training")]})


    def validation_step(self, batch, batch_idx):
        x1, x2, idt, _, _ = batch
        xx = torch.cat([x1,x2],0)

        outputs = self.forward(xx)

        yy, z_latents, zz, z_aggr = \
                    outputs["yy"],\
                    outputs["z_latents"], \
                    outputs["zz"], \
                    outputs["z_aggrs"]

        # Reconstruction loss.
        loss_rec = self.rec_criterion(xx, yy)
        yy = yy.view( *([2,-1] + list(yy.shape[1:])))

        self.log("val_loss", loss_rec.detach().cpu())
        if batch_idx==0 and self.hparams.wandb:
            random_image = random.randint(0, self.hparams.batch_size - 1)     # before it was 0, but it always took the same image (due to the new __getitem__)
            self.logger.experiment.log({"valx1": [wandb.Image((xx[random_image, ...]).cpu().numpy(), caption="valx1"),
            wandb.Image((yy[0, random_image, ...]).detach().cpu().numpy(), caption="valy1")]})
        return {
            "val_loss": loss_rec.detach().cpu()
        }


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        # tensorboard = self.logger[1].experiment
        idx1 = random.randint(0, len(self.val_set) - 1)
        idx2 = random.randint(0, len(self.val_set) - 1)

        # Select only the first images from the two pairs.
        x1 = self.val_set[idx1][0].cuda()
        x2 = self.val_set[idx2][0].cuda()

        topil = transforms.ToPILImage()
        I = self.get_dis_image(torch.stack([x1,x2])).cpu()
        if self.hparams.wandb:
            self.logger.experiment.log({'dis_validation': [
                wandb.Image(topil(I.permute(2,0,1)), caption="dis_validation")]})


    def get_dis_image(self, xx):
        model=self
        img1 = xx[0]
        img2 = xx[1]
        col = img1[:,:5,:]*0+1  # Used to divide images in the grid with white small columns.

        outputs = model.forward(torch.cat([img1[None,...], img2[None,...]], 0))

        rec1 = outputs['yy'][0].detach().clip(0,1)
        rec2 = outputs['yy'][1].detach().clip(0,1)

        I=None
        for i in range(len(outputs["z_latents"])):
            z1 = [x[0]+0 for x in outputs["z_latents"]]     # First image latent factors.
            z2 = [x[1]+0 for x in outputs["z_latents"]]     # Second image latent factors.
            a = torch.linspace(1,0,4)[1:,None].to(xx.device)
            z1 = [z1.repeat(a.shape[0],1) for z1,z2 in zip(z1,z2)]  # Repeat each 1x10 vector 3 times on the rows -> z1 = 10*[3,10].
            z1[i] = z1[i][0,:]*a + (1-a)*z2[i][:]   # Change a factor.

            z = model.aggregate(z1)
            res = model.post_process(model.decoder.forward(z).detach()).clip(0,1)
            res = res.flatten().reshape(*list(res.shape))
            Irow = torch.cat([img1,col,col ,rec1, col]+sum([[r,col] for r in res],[])+ [rec2,col,col,img2],1)
            if I is None:
                I=Irow
            else:
                I = torch.cat([I,Irow],0)
        return I


    def predict_step(self, batch, batch_idx):
        x1, x2, idt, gt1, gt2 = batch
        xx = torch.cat([x1,x2], 0)

        outputs = self.forward(xx)

        yy, z_latents, zz, z_aggr = \
                    outputs["yy"],\
                    outputs["z_latents"], \
                    outputs["zz"], \
                    outputs["z_aggrs"]

        if batch_idx == 0:
            idx1 = random.randint(0, len(self.val_set) - 1)
            idx2 = random.randint(0, len(self.val_set) - 1)
            # Select only the first images from the two pairs.
            x1_rand = self.val_set[idx1][0].cuda()
            x2_rand = self.val_set[idx2][0].cuda()

            I = self.get_dis_image(torch.stack([x1_rand, x2_rand]))
            return {"z_latents": [z_latent.detach().cpu().numpy() for z_latent in z_latents], "image": I.detach().cpu().numpy()}

        return {"z_latents": [z_latent.detach().cpu().numpy() for z_latent in z_latents]}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.hparams.weight_decay)

    def setup(self, stage=None):
        dataset_name = dataset_names[self.hparams.n_dataset]
        print(f'Loading {dataset_name} dataset...')

        transformations = [transforms.ToTensor()]
        if "woof" or "cub" or "cars" or "pets" in dataset_name:
            transformations.append(transforms.Resize(size=(224, 224)))
        if "jaffe" in dataset_name:
            transformations.append(transforms.Resize((64,64)))
        if self.hparams.occlusion:
            transformations.append(transforms.RandomErasing(value=(0,0,0), p=1, scale=(0.1, 0.4)))
        if self.hparams.noise:
            transformations.append(SaltAndPepperNoise(noiseType="SnP"))
        if self.hparams.crop:
            transformations.append(transforms.RandomResizedCrop(64, scale=(0.8, 0.8), ratio=(1., 1.)))
        transform = transforms.Compose(transformations)


        if self.hparams.forced:
            # observation_factor is the only factor over which we have control.
            # oracle_probability indicates the probability that the loader returns a pair in which the observation_factor is different.
            # random_item is true if you want the loader to extract the first image of each pair at random.
            # If false, it is guaranteed that all the images in the datasetwill be seen at least once.
            self.train_set = LimitedDatasetVariableK(dataset_name=dataset_name, seed=self.hparams.seed, factors=None, k=self.hparams.k,
                                                     fraction=self.hparams.dataset_fraction, transform=transform,
                                                     observation_factor=self.hparams.observation,
                                                     oracle_probability=self.hparams.oracle_probability,
                                                     random_item=self.hparams.random_item)
            self.val_set = LimitedDatasetVariableK(dataset_name=dataset_name, seed=self.hparams.seed, factors=None, k=self.hparams.k,
                                                   fraction=self.hparams.dataset_fraction, transform=transform,
                                                   observation_factor=self.hparams.observation,
                                                   oracle_probability=self.hparams.oracle_probability,
                                                   random_item=self.hparams.random_item, split_type="val")
        else:
            self.train_set = BasicDatasetVariableK(dataset_name=dataset_name, seed=self.hparams.seed, factors=None, k=self.hparams.k,
                                                   fraction=self.hparams.dataset_fraction, transform=transform,
                                                   full_random=self.hparams.full_random)

            self.val_set = BasicDatasetVariableK(dataset_name=dataset_name, seed=self.hparams.seed, factors=None, k=self.hparams.k,
                                                 fraction=self.hparams.dataset_fraction, transform=transform,
                                                 full_random=self.hparams.full_random)

        print(f"{dataset_name} - train: {self.train_set.__len__()}, val: {self.val_set.__len__()} - factor sizes {self.train_set.dataset.factors_num_values}")
        print('Done')

        # # For debugging only, reduces the size of the complete dataset(s)
        # dbg_idx = list(range(0, 400))
        # self.train_set = torch.utils.data.Subset(self.train_set, dbg_idx)
        # self.val_set = torch.utils.data.Subset(self.val_set, dbg_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=strtobool(self.hparams.train_shuffle),
            num_workers=0,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # Model specific.
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--n_latent_factors", default=10, type=int)
        parser.add_argument("--latentdim", default=10, type=int)
        parser.add_argument("--in_dim", default=3, type=int)

        parser.add_argument("--n_dataset", default=0, type=int)
        parser.add_argument("--k", default=1, type=int)

        parser.add_argument("--lambda_distribution", default=1e-4, type=float)
        parser.add_argument("--lambda_sparsity", default=1e-1, type=float)
        parser.add_argument("--lambda_consistency", default=1e2, type=float)
        parser.add_argument("--lambda_distance", default=1e-1, type=float)

        # training specific (for this model)
        parser.add_argument("--learning_rate", default=5e-4, type=float)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--weight_decay", default=1e-6, type=float)
        parser.add_argument("--train_shuffle", default="True", type=str)

        return parser
