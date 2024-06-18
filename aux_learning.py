#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from torchvision import transforms as T
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger

from data.aux_datasets import Aux3DShapes, AuxFaces, AuxWoof, AuxCub, AuxCars, AuxCifar, AuxSvhn
from models.aux_learning_model import Detaux


def main(args):
    # Seeds for reproducibility.
    seed_everything(args.seed, workers=True)

    if args.dataset_name in ['woof', 'cub', 'cars']:
        if args.augment:
            print('Using data augmentation...')
            train_transforms = T.Compose([
                T.Resize(size=(256, 256)),
                T.RandomCrop(224),          # Crop to 224x224 a random patch.
                T.RandomHorizontalFlip(),   # Data augmentation.
                T.ToTensor(),               # To tensor and scale to 0-1.
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet statistics.
            ])
            test_transforms = T.Compose([
                T.Resize(size=(256, 256)),
                T.CenterCrop(224),          # At test time we want deterministic cropping.
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet statistics.
            ])
        else:
            train_transforms = T.Compose([
                T.Resize(size=(224, 224)),
                T.ToTensor(),               # To tensor and scale to 0-1.
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet statistics.
            ])
            test_transforms = T.Compose([
                T.Resize(size=(224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet statistics.
            ])
    elif args.dataset_name in ['cifar', 'svhn', 'shapes3d_custom_detangle']:
        train_transforms = T.Compose([
            T.Resize(size=(64, 64)),
            T.ToTensor()
        ])
        test_transforms = train_transforms
    else:
        train_transforms = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor()
        ])
        test_transforms = train_transforms

    print('Loading datasets and dataloaders...')
    if 'shapes' in args.aux_labels:
        trainset = Aux3DShapes(
            mined_labels=args.aux_labels,
            split_type='train',
            noise=args.noise,
            main_task_idx=args.main_task_idx,
            split_factor_idx=[0,1],
            num_test_set_exclusive=[5,5],
            reduced=False,
            dataset_path=args.dataset_path
        )
        testset = Aux3DShapes(
            mined_labels=args.aux_labels,
            split_type='test',
            noise=args.noise,
            main_task_idx=args.main_task_idx,
            split_factor_idx=[0,1],
            num_test_set_exclusive=[5,5],
            reduced=False,
            dataset_path=args.dataset_path
        )
    elif 'faces' in args.aux_labels:
        trainset = AuxFaces(
            data_path = Path(args.data_path),
            mined_labels=args.aux_labels,
            split_type='train',
            transforms=train_transforms,
            main_task_idx=args.main_task_idx
        )
        testset = AuxFaces(
            data_path = Path(args.data_path),
            mined_labels=args.aux_labels,
            split_type='val',
            transforms=test_transforms,
            main_task_idx=args.main_task_idx
        )
    elif 'cub' in args.aux_labels:
        trainset = AuxCub(
            data_path = Path(args.data_path),
            mined_labels=args.aux_labels,
            split_type='train',
            transforms=train_transforms,
            main_task_idx=args.main_task_idx
        )
        testset = AuxCub(
            data_path = Path(args.data_path),
            mined_labels=args.aux_labels,
            split_type='val',
            transforms=test_transforms,
            main_task_idx=args.main_task_idx
        )
    elif 'cars' in args.aux_labels:
        trainset = AuxCars(
            data_path = Path(args.data_path),
            mined_labels=args.aux_labels,
            split_type='train',
            transforms=train_transforms,
            main_task_idx=args.main_task_idx
        )
        testset = AuxCars(
            data_path = Path(args.data_path),
            mined_labels=args.aux_labels,
            split_type='val',
            transforms=test_transforms,
            main_task_idx=args.main_task_idx
        )
    elif 'cifar' in args.aux_labels:
        trainset = AuxCifar(
            data_path = Path(args.data_path),
            mined_labels=args.aux_labels,
            split_type='train',
            transforms=train_transforms
        )
        testset = AuxCifar(
            data_path = Path(args.data_path),
            mined_labels=args.aux_labels,
            split_type='test',
            transforms=test_transforms,
        )
    elif 'svhn' in args.aux_labels:
        trainset = AuxSvhn(
            data_path = Path(args.data_path),
            mined_labels=args.aux_labels,
            split_type='train',
            transforms=train_transforms
        )
        testset = AuxSvhn(
            data_path = Path(args.data_path),
            mined_labels=args.aux_labels,
            split_type='test',
            transforms=test_transforms,
        )

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print('Done!')
    print('Task label sizes:', trainset.task_lbl_sizes, testset.task_lbl_sizes)

    # Output sizes.
    aux_labels = np.load(args.aux_labels)
    aux_out_size = len(np.unique(aux_labels))
    main_out_size = trainset.task_lbl_sizes[0]

    # Build the model.
    num_tasks = len(args.mtl_task_ids)
    model = Detaux(
        hidden_dim=args.hidden_dim,
        num_tasks=num_tasks,
        task_ids=args.mtl_task_ids,
        output_sizes=(main_out_size, aux_out_size),
        dataset_name=args.dataset_name,
        learning_rate=args.learning_rate,
        feature_extractor=args.feature_extractor,
        uncertainty=args.uncertainty
    )

    # Training and evaluation.
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M")
    run_name = f'aux_learning_{args.dataset_name}_task{num_tasks > 1}-mainTaskID{args.main_task_idx}_model{args.feature_extractor}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_path,
        filename=f'{run_name}'+ '{epoch}-' + dt_string,
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        verbose=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"last_{run_name}"

    # Additional information in the run name when logging.
    if args.add_info:
        run_name += f"_{args.add_info}"

    trainer = None
    if args.wandb:
        trainer = Trainer(
            # devices=2,
            # strategy="ddp",
            devices=[args.gpu_num],
            accelerator='gpu',
            max_epochs=args.epochs,
            check_val_every_n_epoch=1,
            callbacks=[checkpoint_callback]
        )
    else:
        wandb_logger = WandbLogger(
            project="", name=run_name, reinit=True, entity=''
        )
        trainer = Trainer(
            # devices=2,
            # strategy="ddp",
            devices=[args.gpu_num],
            accelerator='gpu',
            max_epochs=args.epochs,
            check_val_every_n_epoch=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback]
        )
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detaux: Auxiliary Learning.')
    parser.add_argument('--aux_labels', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--dataset_name', type=str, default='cub')
    parser.add_argument('--main_task_idx', type=int, default=2)
    parser.add_argument('--mtl_task_ids', nargs='+', type=int, default=[0])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--use_nddr', action='store_true')
    parser.add_argument('--use_mti', action='store_true')
    parser.add_argument('--add_info', type=str)
    parser.add_argument('--feature_extractor', type=str, default='vgg')
    parser.add_argument('--uncertainty', action='store_true')
    parser.add_argument('--wandb', action='store_false')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')

    args = parser.parse_args()
    main(args)
