#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import wandb

from datetime import datetime
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from models import conv_models
from models.resnet_ae import get_configs
from models.dis_large_model import LargeModel
from models.dis_basic_model import BasicModel
from data.dis_datasets import dataset_names


def train(args):
    print(f'Running with arguments: {args}')

    # Seeds for reproducibility.
    seed = args.seed
    seed_everything(seed, workers=True)

    # Create run_name.
    dt_string = datetime.now().strftime('%d-%m-%Y-%H-%M')
    run_name = f'DIS_{dataset_names[args.n_dataset]}_BS{args.batch_size}_E{args.max_epochs}_NLFactors{args.n_latent_factors}_LatentDim{args.latentdim}'

    if args.k > 0:
        run_name += f'_k{args.k}'
    else:
        run_name += '_kall'

    if args.occlusion:
        run_name += f'_occlusion'

    if args.noise:
        run_name += f'_noise'

    if args.crop:
        run_name += f'_crop'

    if args.observation >= 0:
        run_name += f'_obs{args.observation}'

    if args.oracle_probability:
        run_name += f'_op{args.oracle_probability}'

    if args.random_item:
        run_name += f'_randomitem'

    if args.full_random:
        run_name += f'_fullrandom'

    # Additional information in the run name.
    if args.add_info:
        run_name += f'_{args.add_info}'

    if args.forced:
        assert args.observation >= 0 and args.oracle_probability
        run_name += f'_forced'

    run_name += f'_{dt_string}'

    # TODO: Check if this is correct.
    if args.ckpt:
        run_name = os.path.basename(args.ckpt)[:-5]

    if args.wandb:
        wandb_logger = WandbLogger(
            project="",
            name=run_name,
            reinit=True,
            entity=''
        )

    # Load autoencoder architecture.
    if args.conv_model == 'resnetae':
        resnet_configs, _ = get_configs("resnet18")
        encoder_args = [resnet_configs, args.latentdim]
        decoder_args = [resnet_configs[::-1], args.latentdim]
    else:
        encoder_args = [args.latentdim, args.in_dim, 64, args.batch_size, False]
        decoder_args = [args.latentdim, args.in_dim, 64, False]

    # 8 = ImageWoof, 13 = CUB, 14 = MEDIC, 15 = Cars, 16 = Pets, 17 = CIFAR, 18 = SVHN.
    if args.n_dataset in [8, 13, 14, 15, 16, 17, 18]:
        # AE w/o FC bottleneck.
        model = LargeModel(
            hparams=args,
            encoder=conv_models[args.conv_model][0](*encoder_args, final_fc=False),
            decoder=conv_models[args.conv_model][1](*decoder_args, initial_fc=False)
        )
    else:
        # AE w/ FC bottleneck.
        model = BasicModel(
            hparams=args,
            encoder=conv_models[args.conv_model][0](*encoder_args),
            decoder=conv_models[args.conv_model][1](*decoder_args)
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.save_dir,
        filename=run_name + "_{epoch}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        verbose=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"last_{run_name}"

    if args.wandb:
        # Without 16 bit precision.
        trainer = Trainer.from_argparse_args(args, devices=[args.gpu_idx], accelerator="gpu", logger=wandb_logger, callbacks=[checkpoint_callback], deterministic=True)
    elif args.save_dir:
        # Without 16 bit precision.
        trainer = Trainer.from_argparse_args(args, devices=[args.gpu_idx], accelerator="gpu", callbacks=[checkpoint_callback], deterministic=True)
    else:
        trainer = Trainer.from_argparse_args(args, devices=[args.gpu_idx], accelerator="gpu", deterministic=True)

    if not args.ckpt:
        trainer.fit(model)
        output = trainer.predict(ckpt_path=trainer.checkpoint_callback.last_model_path)
    else:
        output = trainer.predict(model, ckpt_path=args.ckpt)

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(description='Detaux: Disentanglement.')
    parser = Trainer.add_argparse_args(parser)

    parser.set_defaults(max_epochs=-1)
    parser.set_defaults(check_val_every_n_epoch=1)
    parser = LargeModel.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, help='seed for randoms', default=1234)
    parser.add_argument('--dataset_path', type=str)

    parser.add_argument('--warmup_c', action='store_true')
    parser.add_argument('--latattn', action='store_true')
    parser.add_argument('--contrast', action='store_true')
    parser.add_argument('--dataset_fraction', type=float, default=1.0)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[])
    parser.add_argument('--conv_model', type=str, choices=list(conv_models.keys()), default='simpleconv')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--add_info', type=str)
    parser.add_argument('--occlusion', action='store_true')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--full_random', action='store_true')
    parser.add_argument('--observation', type=int, default=-1)
    parser.add_argument('--oracle_probability', type=float)
    parser.add_argument('--random_item', action='store_true')
    parser.add_argument('--outpath', type=str, default='latent_spaces')
    parser.add_argument('--ckpt', type=str, help='if given, skip training and try testing with this checkpoint')

    # Store false for sweep (tmp).
    parser.add_argument('--forced', action='store_true', help='if True, use the forced disentanglement model with the modified oracle')
    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()
    train(args)
