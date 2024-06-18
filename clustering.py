#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import torch
import hdbscan

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.cluster import KMeans
from argparse import ArgumentParser
from sklearn.decomposition import PCA
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import completeness_score, homogeneity_score

from models import conv_models
from models.resnet_ae import get_configs
from models.dis_basic_model import BasicModel
from models.dis_large_model import LargeModel
from data.dis_datasets import dataset_names


def run(args):
    start_time = time.time()
    print("Running with arguments:")
    print(args)

    # Seeds for reproducibility.
    seed = args.seed
    dataset_name = dataset_names[args.n_dataset]
    seed_everything(seed, workers=True)

    if args.conv_model == "resnetae":
        resnet_configs, _ = get_configs("resnet18")
        encoder_args = [resnet_configs, args.latentdim]
        decoder_args = [resnet_configs[::-1], args.latentdim]
    else:
        encoder_args = [args.latentdim, args.in_dim, 64, args.batch_size, False]
        decoder_args = [args.latentdim, args.in_dim, 64, False]

    if (
        "cub" not in dataset_name
        and "woof" not in dataset_name
        and "medic" not in dataset_name
        and "cars" not in dataset_name
    ):
        disentanglement_model = BasicModel(
            hparams=args,
            encoder=conv_models[args.conv_model][0](*encoder_args),
            decoder=conv_models[args.conv_model][1](*decoder_args),
        )
    else:
        disentanglement_model = LargeModel(
            hparams=args,
            encoder=conv_models[args.conv_model][0](*encoder_args, final_fc=False),
            decoder=conv_models[args.conv_model][1](
                *decoder_args, initial_fc=False
            ),
        )

    disentanglement_model.load_state_dict(
        torch.load(
            args.detangle_ckpt,
            map_location=lambda storage, loc: storage.cuda(args.gpu_idx),
        )["state_dict"],
        strict=True,
    )
    disentanglement_model.to(f"cuda:{args.gpu_idx}")
    disentanglement_model.eval()

    # Prepare datasets and loaders.
    disentanglement_model.setup()

    if "shapes" in args.detangle_ckpt:
        dataset = disentanglement_model.train_set
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=4
        )
        trainloader = loader
    else:
        trainset = disentanglement_model.train_set
        testset = disentanglement_model.val_set
        dataset = torch.utils.data.ConcatDataset([trainset, testset])
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=32, shuffle=False, num_workers=4
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=4
        )

    # Stage 1: Choose which subspace to use for the auxilliary loss clustering procedure.
    # We pick the subspace that is most disentangled according to the main losses: consistency and distance.
    if args.subspace != -1:
        best_subspace = args.subspace
    else:
        dotprods = []
        with torch.no_grad():
            for batch in tqdm(
                trainloader, desc="Calculating best subspace to cluster..."
            ):
                x1, x2, _, gt1, gt2 = batch
                (
                    x1,
                    x2,
                ) = x1.to(
                    f"cuda:{args.gpu_idx}"
                ), x2.to(f"cuda:{args.gpu_idx}")
                gt1, gt2 = gt1.to(f"cuda:{args.gpu_idx}"), gt2.to(
                    f"cuda:{args.gpu_idx}"
                )

                xx = torch.cat([x1, x2], 0)
                outputs = disentanglement_model(xx)

                _, z_latents, _, _ = (
                    outputs["yy"],
                    outputs["z_latents"],
                    outputs["zz"],
                    outputs["z_aggrs"],
                )

                # Split in z1 and z2 in the first dimension.
                z_latents = (
                    torch.stack(z_latents, 0)
                    .view(len(z_latents), 2, -1, z_latents[0].shape[-1])
                    .transpose(0, 1)
                )

                vote_latents = z_latents[0]
                dotprod = torch.bmm(
                    # Compute dot product of the representation in the forced subspace with all the others.
                    vote_latents[1:].permute(1, 0, 2), vote_latents[0].unsqueeze(-1)
                ).squeeze()
                dotprods.append(dotprod)

                # mis = []
                # for sub in z_latents[0]:
                #     if len(gt1.shape) > 1:
                #         gt1 = gt1[:, args.observation]

                #     avg_sub_mi = mutual_info_classif(sub.detach().cpu().numpy(),
                #                                     gt1.detach().cpu().numpy(),
                #                                     n_neighbors=3, # Faces
                #                                     random_state=args.seed).mean()
                #     mis.append(avg_sub_mi)
                # best_subspaces.append(np.argmin(mis))

    dotprods = torch.vstack(dotprods)
    # Have to add 1 since we're computing the dot product of the representation in the other k-1 subspaces.
    best_subspace = dotprods.mean(0).abs().argsort()[0].item() + 1
    print(f"Best subspace is: #{best_subspace}")

    # Stage 2: Clustering on the latent space of the best subspace.
    print(f"Starting to cluster features of subspace #{best_subspace}")
    X, y = [], []
    with torch.no_grad():
        for batch in tqdm(
            loader, desc="Extracting latent spaces i.e., clustering feature spaces"
        ):
            torch.cuda.empty_cache()
            x, _, _, gt, _ = batch
            x = x.to(f"cuda:{args.gpu_idx}")
            x = disentanglement_model.pre_process(x)
            z = disentanglement_model.encoder(x)

            if args.rec_only:
                X.append(z.detach().cpu())
            else:
                z_latents = disentanglement_model.latent_encode(z)
                # Turn feature maps into flat vector representations.
                if (
                    "cub" in dataset_name
                    or "woof" in dataset_name
                    or "medic" in dataset_name
                    or "cars" in dataset_name
                ):
                    z_latents = [x.flatten(1) for x in z_latents]
                X.append(z_latents[best_subspace].detach().cpu())
            y.append(gt.detach().cpu())

    X = torch.vstack(X).numpy()
    if "faces" in dataset_name:
        y_main_task_idx = [y[i][:, 2] for i in range(len(y))]
        y = torch.hstack(y_main_task_idx).numpy()
    else:
        y = torch.hstack([l.flatten() for l in y]).numpy()

    # Prepare data for clustering.
    if args.reduce_dim:
        print("Reducing cluster dimension via PCA...")
        reducer = PCA(n_components=256).fit(X)
        print("Explained variance by PCA:", reducer.explained_variance_ratio_.sum())
        X = reducer.transform(X)

    if "shapes3d" in args.detangle_ckpt:
        rng = np.random.default_rng()
        cluster_train_idx = rng.choice(X.shape[0], size=50000, replace=False)
        X_train = X[cluster_train_idx]
    else:
        X_train = X[: len(trainset)]

    # Clustering.
    print(f"Starting clustering procedure using {args.clustering_method}...")
    start_time = time.time()
    if args.num_clusters != -1:
        cluster_module = KMeans(
            n_clusters=args.num_clusters, random_state=seed, max_iter=1000, verbose=1
        ).fit(
            X_train
        )
        # scikit-learn 1.0.2
        # cluster_module = KMeans(n_clusters=args.num_clusters, random_state=seed, n_init='auto', max_iter=1000).fit(X_train)   # scikit-learn 1.2.0
    else:
        # min_cluster_size  = int(len(X_train) * 0.01)  # Initial Faces == 16
        # min_cluster_size  = int(len(X_train) * 0.001) # Initial Woof == 12

        # min_cluster_size = 15   # Faces.
        # min_cluster_size = 200  # Cars.
        min_cluster_size = 200    # Shapes3D.

        cluster_module = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, prediction_data=True
        ).fit(X_train)

    if args.num_clusters != -1:
        cluster_labels = cluster_module.predict(X)
    else:
        cluster_labels, _ = hdbscan.prediction.approximate_predict(cluster_module, X)

    if args.train_only:
        X = X[: len(trainset)]
        y = y[: len(trainset)]
        cluster_labels = cluster_labels[: len(trainset)]
    elif args.test_only:
        X = X[-len(testset) :]
        y = y[-len(testset) :]
        cluster_labels = cluster_labels[-len(testset) :]

    if -1 in cluster_labels:
        print(
            "Clustering finished with -1 labels. This idicates noisy data points and non-convergence."
        )
        cluster_labels += 1

    # Print out information.
    print(
        f"Completed {cluster_module.__class__.__name__} clustering after {time.time() - start_time} seconds."
    )
    unique_lbls, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Number of clusters is: {len(unique_lbls)}")
    print("Label counts:")
    print(unique_lbls)
    print(counts)

    # Visualize clustering results in low dim.
    # if len(unique_lbls) > 1:
    #     if not os.path.exists("outputs"):
    #         os.makedirs("outputs")
    #     if X.shape[1] > 2:
    #         fig = plt.figure()
    #         ax = fig.add_subplot(projection="3d")
    #         space3d = PCA(n_components=3).fit_transform(X)
    #         ax.scatter(space3d[:, 0], space3d[:, 1], space3d[:, 2], c=cluster_labels)
    #         plt.savefig(
    #             f"outputs/3DPCA_dataset#{disentanglement_model.hparams.n_dataset}_subspace#{best_subspace}_trainOnly#{args.train_only}_testOnly#{args.test_only}_{args.clustering_method}.png"
    #         )
    #         plt.close()
    #     else:
    #         plt.scatter(X[:, 0], X[:, 1], c=cluster_labels)
    #         plt.savefig(
    #             f"outputs/2DPCA_dataset#{disentanglement_model.hparams.n_dataset}_subspace#{best_subspace}_trainOnly#{args.train_only}_testOnly#{args.test_only}_{args.clustering_method}.png"
    #         )
    #         plt.close()

    # Calculate clustering completeness and homogeniety if operating on the principal task.
    # if len(unique_lbls) > 1:
    #     if args.subspace == 0:
    #         homogeneity = homogeneity_score(y, cluster_labels)
    #         completeness = completeness_score(y, cluster_labels)
    #         print(f"H-score w.r.t principal task labels: {homogeneity}")
    #         print(f"C-score w.r.t principal task labels: {completeness}")

    # Save labels to file.
    if not os.path.exists("outputs/aux_labels"):
        os.makedirs("outputs/aux_labels")

    lbls_name = (
        "_".join(args.detangle_ckpt.split("/")[-1].split("_")[1:])[:-5] + "_labels"
    )
    save_name = f"{lbls_name}_subspace_#{best_subspace}_trainOnly#{args.train_only}_testOnly#{args.test_only}_clustering_method{args.clustering_method}.npy"
    np.save(os.path.join("outputs/aux_labels", save_name), cluster_labels)
    print(f"Exported latent labels to {save_name}")

    print("Clustering procedure done!")
    print("Total runtime: --- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    parser = ArgumentParser(description="Detaux: Clustering.")
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(check_val_every_n_epoch=1)
    parser = BasicModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, help="seed for randoms", default=1234)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--random_probability", type=float, default=0.0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--warmup_c", action="store_true")
    parser.add_argument("--latattn", action="store_true")
    parser.add_argument("--contrast", action="store_true")
    parser.add_argument("--dataset_fraction", type=float, default=1.0)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="dir for saving .ckpt models",
    )
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[])
    parser.add_argument(
        "--conv_model", type=str, choices=list(conv_models.keys()), default="simpleconv"
    )
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--add_info", type=str)
    parser.add_argument("--occlusion", action="store_true")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--full_random", action="store_true")
    parser.add_argument("--observation", type=int, default=-1)
    parser.add_argument("--oracle_probability", type=float)
    parser.add_argument("--random_item", action="store_true")
    parser.add_argument("--outpath", type=str, default="latent_spaces")
    parser.add_argument(
        "--ckpt",
        type=str,
        help="if given, skip training and try testing with this checkpoint",
    )
    parser.add_argument(
        "--forced",
        action="store_true",
        help="if True, use the forced disentanglement model with the modified oracle",
    )

    parser.add_argument(
        "--detangle_ckpt",
        type=str,
        required=True,
        help="trained model checkpoint for disentanglement layers",
    )
    parser.add_argument("--clustering_method", type=str, default="HDB")
    parser.add_argument("--reduce_dim", action="store_true")

    # New params.
    parser.add_argument("--subspace", type=int, default=-1)
    parser.add_argument("--num_clusters", type=int, default=-1)
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--rec_only", action="store_true")

    args = parser.parse_args()
    run(args)
