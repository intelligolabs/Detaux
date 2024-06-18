#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import pytorch_lightning as pl
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score
from torchvision.models import vgg16, vgg16_bn, resnet18, swin_b
from torchvision.models import VGG16_Weights, VGG16_BN_Weights, ResNet18_Weights, Swin_B_Weights


def auxilearn_focal_loss(x_pred, x_output, num_output):
    """Focal loss
    :param x_pred:  prediction of primary network (either main or auxiliary)
    :param x_output: label
    :param num_output: number of classes
    :return: loss per sample
    """

    x_output_onehot = torch.zeros((len(x_output), num_output)).to(x_pred.device)
    x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)
    x_pred = F.softmax(x_pred, dim=1)

    loss = torch.sum(- x_output_onehot * (1 - x_pred) ** 2 * torch.log(x_pred + 1e-12), dim=1).mean()

    return loss


class Detaux(pl.LightningModule):
    def __init__(
            self,
            hidden_dim,
            num_tasks,
            task_ids,
            output_sizes,
            dataset_name,
            learning_rate=1e-4,
            feature_extractor='vgg',
            uncertainty=False
        ):
        super(Detaux, self).__init__()
        self.save_hyperparameters()

        pretrained_weights = None
        if dataset_name in ['woof', 'cub', 'cars']:
            if feature_extractor == 'vgg':
                pretrained_weights = VGG16_Weights.IMAGENET1K_V1
            elif feature_extractor == 'vgg_bn':
                pretrained_weights = VGG16_BN_Weights.IMAGENET1K_V1
            elif feature_extractor == 'swin':
                pretrained_weights = Swin_B_Weights.IMAGENET1K_V1
            else:
                pretrained_weights = ResNet18_Weights.IMAGENET1K_V1

        if feature_extractor == 'vgg':
            self.feature_extractor = vgg16(weights=pretrained_weights).features
        elif feature_extractor == 'vgg_bn':
            self.feature_extractor = vgg16_bn(weights=pretrained_weights).features
        elif feature_extractor == 'swin':
            self.feature_extractor = swin_b(weights=pretrained_weights).features
        else:
            resnet = resnet18(weights=pretrained_weights)
            modules = list(resnet.children())[:-2]
            self.feature_extractor = nn.Sequential(*modules)

        self.dataset_name = dataset_name
        self.num_tasks = num_tasks
        self.task_ids = task_ids
        self.learning_rate = learning_rate

        self.classification_heads = nn.ModuleList()
        if feature_extractor == 'swin':
            classhead_in_dim = 1024
        else:
            classhead_in_dim = 512

        for task_id in task_ids:
                self.classification_heads.append(
                    nn.Sequential(
                        nn.Linear(classhead_in_dim*2*2, classhead_in_dim),     # 64x64 img.
                        # nn.Linear(classhead_in_dim*7*7, classhead_in_dim),   # 224x224 img.
                        nn.Dropout(0.1),
                        nn.ReLU(),
                        nn.Linear(classhead_in_dim, output_sizes[task_id])
                    )
                )

        self.uncertainty = uncertainty
        if self.uncertainty:
            self.log_var_list = nn.Parameter(torch.zeros((self.num_tasks,), requires_grad=True))

    def forward(self, x):
        # Common features from CNN.
        features = self.feature_extractor(x)
        features = torch.flatten(features, start_dim=1)

        task_logits = []
        for i in range(self.num_tasks):
            logits_ti = self.classification_heads[i](features)
            task_logits.append(logits_ti)

        return task_logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=5e-3
        )

        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.trainer.max_epochs//2, gamma=0.1)
        #return [optimizer], [scheduler]

        return optimizer

    def on_train_epoch_start(self):
        self.train_pred = []
        self.train_gt = []

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        losses = []
        for i, task in enumerate(self.task_ids):
            lgs, gt = logits[i], y[:, task]

            # if self.dataset_name in ['cub', 'cars']:
            #     loss = auxilearn_focal_loss(lgs, gt, num_output=lgs.shape[1])
            # else:
            loss = F.cross_entropy(lgs, gt)

            self.log(f'train_task{task}_loss', loss, sync_dist=True)
            losses.append(loss)

        # if self.uncertainty:
        #     for i, task in enumerate(self.task_ids):
        #         self.log(f'train_task{task}_log_var', self.log_var_list[i], sync_dist=True)
        #         losses[i] = losses[i] * torch.exp(-self.log_var_list[i]) + self.log_var_list[i]
        # mtl_loss = sum(losses)

        # mtl_loss = (10*losses[0]) + losses[1]

        mtl_loss = losses[0] + losses[1]
        self.log('train_loss', mtl_loss, sync_dist=True)

        # Save for evaluation.
        self.train_pred.append(logits)
        self.train_gt.append(y)

        # Log learning rate for monitoring.
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, on_step=True, sync_dist=True)

        return mtl_loss

    def on_train_epoch_end(self):
        self.mtl_evaluation(self.train_pred, self.train_gt, self.training)

    def on_validation_epoch_start(self):
        self.val_pred = []
        self.val_gt = []

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        losses = []
        for i, task in enumerate(self.task_ids):
            lgs, gt = logits[i], y[:, task]
            # if self.dataset_name in ['cub', 'cars']:
            #     loss = auxilearn_focal_loss(lgs, gt, num_output=lgs.shape[1])
            # else:
            loss = F.cross_entropy(lgs, gt)
            self.log(f'val_task{i}_loss', loss, sync_dist=True)
            losses.append(loss)
        mtl_loss = sum(losses)
        self.log('val_loss', mtl_loss, sync_dist=True)

        # Save for evaluation.
        self.val_pred.append(logits)
        self.val_gt.append(y)

    def on_validation_epoch_end(self):
        self.mtl_evaluation(self.val_pred, self.val_gt, self.training)

    def mtl_evaluation(self, pred_list, gt_list, training):
        train_str = 'train' if training else 'val'
        for i, task in enumerate(self.task_ids):
            pred, gt = torch.vstack([x[i] for x in pred_list]), torch.hstack([y[:, task] for y in gt_list])
            pred, gt = pred.argmax(-1).detach().cpu().numpy().flatten(), gt.detach().cpu().numpy().flatten()
            acc = accuracy_score(gt, pred)
            f1 = f1_score(gt, pred, average='macro')
            self.log(f'{train_str}_task{task}_accuracy', acc, sync_dist=True)
            self.log(f'{train_str}_task{task}_f1', f1, sync_dist=True)
            if train_str == 'val':
                print(f'Task: {task}, Acc: {acc}, F1: {f1}')
