import argparse
import os
import json
import shutil
import random
from itertools import islice

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss

from data import HistoDataNorm
from model import Glow

import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim

import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms

from PIL import Image



def check_manual_seed(seed):
    #seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("Using seed: {seed}".format(seed=seed))


def check_dataset(dataset, augmentation, missing):
    if dataset == "malaria":
        print("malaria")

        domain_list_train = os.listdir('dataset_sorted_by_domain/')

        dataset = HistoDataNorm('dataset_sorted_by_domain/', domain_list=domain_list_train, augmentation=augmentation)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        # train_dataset = HistoDataNorm('train/', domain_list=domain_list_train, augmentation=False)
        # test_dataset = HistoDataNorm('test/', domain_list=domain_list_train, augmentation=False)

    elif dataset == "malaria_10_domains":
        print("malaria 2 domains")

        #domain_list_train = ['C184P145ThinF', 'C59P20thinF']
        domain_list_train = ["C116P77ThinF", "C132P93ThinF", "C137P98ThinF", "C180P141NThinF", "C182P143NThinF", \
                             "C184P145ThinF", "C39P4thinF", 'C59P20thinF', "C68P29N", "C99P60ThinF"]

        # for domain in domain_list_train:
        #     print("domain length", domain, len(os.listdir('dataset_sorted_by_domain/'+domain + '/Uninfected')+os.listdir('dataset_sorted_by_domain/'+domain + '/Parasitized')))

        # train_dataset = HistoDataNorm('train/', domain_list=domain_list_train, augmentation=False)
        # test_dataset = HistoDataNorm('test/', domain_list=domain_list_train, augmentation=False)

        dataset = HistoDataNorm('dataset_sorted_by_domain/', domain_list=domain_list_train, augmentation=augmentation)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    elif dataset == "missing_Uninfected":
        print("missing_uninfected")

        domain_list_train_1 = ["C116P77ThinF", "C132P93ThinF", "C137P98ThinF", "C180P141NThinF", "C182P143NThinF", \
                             "C184P145ThinF", "C39P4thinF", 'C59P20thinF', "C68P29N", "C99P60ThinF"]                     
        domain_list_train_1.remove(args.missing)
        domain_list_train_1.append(args.missing +'_empty')
        dataset_1 = HistoDataNorm('missing_Uninfected/', domain_list=domain_list_train_1, augmentation=augmentation)
        train_size_1 = int(0.8 * len(dataset_1))
        test_size_1 = len(dataset_1) - train_size_1
        train_dataset_1, test_dataset_1 = torch.utils.data.random_split(dataset_1, [train_size_1, test_size_1])
        
        domain_list_train_2 = ["C116P77ThinF_empty", "C132P93ThinF_empty", "C137P98ThinF_empty", "C180P141NThinF_empty", "C182P143NThinF_empty", \
                             "C184P145ThinF_empty", "C39P4thinF_empty", 'C59P20thinF_empty', "C68P29N_empty", "C99P60ThinF_empty"]   
        domain_list_train_2.remove(args.missing+'_empty')
        domain_list_train_2.append(args.missing +'_m')
        dataset_2 = HistoDataNorm('missing_Uninfected/', domain_list=domain_list_train_2, augmentation=augmentation)
        train_size_2 = int(0.8 * len(dataset_2))
        test_size_2 = len(dataset_2) - train_size_2
        train_dataset_2, test_dataset_2 = torch.utils.data.random_split(dataset_2, [train_size_2, test_size_2])
        
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_1,train_dataset_2])
        test_dataset = torch.utils.data.ConcatDataset([test_dataset_1,test_dataset_2])
        
    
    elif dataset == "missing_Parasitized":
        print("missing_parasitized")

        domain_list_train_1 = ["C116P77ThinF", "C132P93ThinF", "C137P98ThinF", "C180P141NThinF", "C182P143NThinF", \
                             "C184P145ThinF", "C39P4thinF", 'C59P20thinF', "C68P29N", "C99P60ThinF"]                     
        domain_list_train_1.remove(args.missing)
        domain_list_train_1.append(args.missing +'_empty')
        domain_list_train_1.sort()
        dataset_1 = HistoDataNorm('missing_Parasitized/', domain_list=domain_list_train_1, augmentation=augmentation)
        train_size_1 = int(0.8 * len(dataset_1))
        test_size_1 = len(dataset_1) - train_size_1
        train_dataset_1, test_dataset_1 = torch.utils.data.random_split(dataset_1, [train_size_1, test_size_1])
        
        domain_list_train_2 = ["C116P77ThinF_empty", "C132P93ThinF_empty", "C137P98ThinF_empty", "C180P141NThinF_empty", "C182P143NThinF_empty", \
                             "C184P145ThinF_empty", "C39P4thinF_empty", 'C59P20thinF_empty', "C68P29N_empty", "C99P60ThinF_empty"]   
        domain_list_train_2.remove(args.missing+'_empty')
        domain_list_train_2.append(args.missing +'_m')
        domain_list_train_2.sort()
        dataset_2 = HistoDataNorm('missing_Parasitized/', domain_list=domain_list_train_2, augmentation=augmentation)
        train_size_2 = int(0.8 * len(dataset_2))
        test_size_2 = len(dataset_2) - train_size_2
        train_dataset_2, test_dataset_2 = torch.utils.data.random_split(dataset_2, [train_size_2, test_size_2])
        print(len(train_dataset_1))
        print(len(train_dataset_2))
        print(len(test_dataset_1))
        print(len(test_dataset_2))
        
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_1,train_dataset_2])
        test_dataset = torch.utils.data.ConcatDataset([test_dataset_1,test_dataset_2])

    return train_dataset, test_dataset
    


def compute_loss(nll, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    if multi_class:
        #print("multi class?")
        # y_logits = torch.sigmoid(y_logits)
        loss_classes = F.binary_cross_entropy_with_logits(
            y_logits, y, reduction=reduction
        )
    else:
        # print(y_logits)
        loss_classes = F.cross_entropy(
            y_logits, torch.argmax(y, dim=1), reduction=reduction
        )

    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes

    return losses

def compute_loss_yd(nll, y_logits, y_weight, y, d_logits, d_weight, d, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    loss_classes = F.cross_entropy(
            y_logits, torch.argmax(y, dim=1), reduction=reduction)

    loss_domains = F.cross_entropy(
            d_logits, torch.argmax(d, dim=1), reduction=reduction)

    
    losses["loss_classes"] = loss_classes
    losses["loss_domains"] = loss_domains

    losses["total_loss"] = losses["nll"] + y_weight * loss_classes + d_weight*loss_domains
    

    return losses


def main(
    dataset,
    augment,
    batch_size,
    eval_batch_size,
    epochs,
    saved_model,
    seed,
    hidden_channels,
    K,
    L,
    actnorm_scale,
    flow_permutation,
    flow_coupling,
    LU_decomposed,
    learn_top,
    y_condition,
    extra_condition,
    sp_condition,
    d_condition,
    yd_condition,
    y_weight,
    d_weight,
    max_grad_clip,
    max_grad_norm,
    lr,
    n_workers,
    cuda,
    n_init_batches,
    output_dir,
    missing,
    saved_optimizer,
    warmup,
):

    print(output_dir)
    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"
    print(device)
    check_manual_seed(seed)
    print("augmenting?", augment)
    train_dataset, test_dataset = check_dataset(dataset, augment,missing)
    image_shape = (32, 32, 3)


    multi_class = False

    if yd_condition:
        num_classes = 2
        num_domains=10
        #num_classes = 10+2
        #multi_class=True
    elif d_condition:
        num_classes=10
        num_domains=0
    else:
        num_classes=2
        num_domains=0
    #print("num classes", num_classes)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        drop_last=True,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=n_workers,
        drop_last=False,
    )

    model = Glow(
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        num_classes,
        num_domains,
        learn_top,
        y_condition,
        extra_condition,
        sp_condition,
        d_condition,
        yd_condition
    )



    model = model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=5e-5)

    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup)  # noqa
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y, d, yd = batch
        x = x.to(device)

        if y_condition:
            y = y.to(device)
            z, nll, y_logits, spare = model(x, y)
            losses = compute_loss_y(nll, y_logits, y_weight, y, multi_class)
        elif d_condition:
            d = d.to(device)
            z, nll, d_logits, spare = model(x, d)

            losses = compute_loss_y(nll, d_logits, d_weight, d, multi_class)
        elif yd_condition:
            y, d, yd = y.to(device), d.to(device), yd.to(device)
            z, nll, y_logits, d_logits = model(x, yd)
            losses = compute_loss_yd(nll, y_logits, y_weight, y, d_logits, d_weight, d)
        else:
            print("none")
            z, nll, y_logits, spare = model(x, None)
            losses = compute_loss(nll)

        losses["total_loss"].backward()

        if max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        return losses

    def eval_step(engine, batch):
        model.eval()

        x, y, d, yd = batch
        x = x.to(device)

        with torch.no_grad():
            if y_condition:
                y = y.to(device)
                z, nll, y_logits, none_logits = model(x, y)
                losses = compute_loss_y(
                    nll, y_logits, y_weight, y, multi_class, reduction="none"
                )
            elif d_condition:
                d = d.to(device)
                z, nll, d_logits, non_logits = model(x, d)
                losses = compute_loss_y(
                    nll, d_logits, d_weight, d, multi_class, reduction="none"
                )
            elif yd_condition:
                y, d, yd = y.to(device), d.to(device), yd.to(device)
                z, nll, y_logits, d_logits = model(x, yd)
                losses = compute_loss_yd(
                    nll, y_logits, y_weight, y, d_logits, d_weight, d, reduction="none"
                )
            else:

                z, nll, y_logits, d_logits = model(x, None)
                losses = compute_loss(nll, reduction="none")
        #print(losses, "losssssess")
        return losses

    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(
        output_dir, "glow", save_interval=1, n_saved=2, require_empty=False
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        checkpoint_handler,
        {"model": model, "optimizer": optimizer},
    )


    monitoring_metrics = ["total_loss"]
    RunningAverage(output_transform=lambda x: x["total_loss"]).attach(
        trainer, "total_loss"
    )

    evaluator = Engine(eval_step)
    
    # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
    Loss(
        lambda x, y: torch.mean(x),
        output_transform=lambda x: (
            x["total_loss"],
            torch.empty(x["total_loss"].shape[0]),
        ),
    ).attach(evaluator, "total_loss")

    if y_condition or d_condition or yd_condition:
        monitoring_metrics.extend(["nll"])
        RunningAverage(output_transform=lambda x: x["nll"]).attach(trainer, "nll")

        # Note: replace by https://github.com/pytorch/ignite/pull/524 when released
        Loss(
            lambda x, y: torch.mean(x),
            output_transform=lambda x: (x["nll"], torch.empty(x["nll"].shape[0])),
        ).attach(evaluator, "nll")


    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    # load pre-trained model if given
    if saved_model:
        model.load_state_dict(torch.load(saved_model))
        model.set_actnorm_init()

        if saved_optimizer:
            optimizer.load_state_dict(torch.load(saved_optimizer))

        file_name, ext = os.path.splitext(saved_model)
        resume_epoch = int(file_name.split("_")[-1])

        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.epoch = resume_epoch
            engine.state.iteration = resume_epoch * len(engine.state.dataloader)

    @trainer.on(Events.STARTED)
    def init(engine):
        model.train()

        init_batches = []
        init_targets = []
        init_domains = []
        init_yds = []

        with torch.no_grad():
            for batch, target, domain, yd in islice(train_loader, None, n_init_batches):
                init_batches.append(batch)
                init_targets.append(target)
                init_domains.append(domain)
                init_yds.append(yd)


            init_batches = torch.cat(init_batches).to(device)

            assert init_batches.shape[0] == n_init_batches * batch_size

            if y_condition :
                init_targets = torch.cat(init_targets).to(device)
                model(init_batches, init_targets)
            elif d_condition:
                init_domains = torch.cat(init_domains).to(device)
                model(init_batches, init_domains)
            elif yd_condition:
                init_yds = torch.cat(init_yds).to(device)
                model(init_batches, init_yds)
            else:
                init_targets = None
                model(init_batches, init_targets)


    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(engine):
        evaluator.run(test_loader)
        #print("done")
        scheduler.step()
        metrics = evaluator.state.metrics

        losses = ", ".join([f"{key}: {value:.8f}" for key, value in metrics.items()])

        print(f"Validation Results - Epoch: {engine.state.epoch} {losses}")


    def score_function(engine):
        val_loss = engine.state.metrics['total_loss']

        return -val_loss
        
        
    name = "best_" 

    val_handler = ModelCheckpoint(
        output_dir, name, score_function=score_function, score_name="val_loss", n_saved=1, require_empty=False
    )

    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED,
        val_handler,
        {"model": model},
    )

    
    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message(
            f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]"
        )
        timer.reset()

    trainer.run(train_loader, epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="malaria",
        help="Type of the dataset to be used.",
    )

    parser.add_argument(
        "--no_augment",
        action="store_false",
        dest="augment",
        help="Augment training data",
    )

    parser.add_argument(
        "--hidden_channels", type=int, default=512, help="Number of hidden channels"
    )

    parser.add_argument("--K", type=int, default=32, help="Number of layers per block")

    parser.add_argument("--L", type=int, default=3, help="Number of blocks")

    parser.add_argument(
        "--actnorm_scale", type=float, default=1.0, help="Act norm scale"
    )

    parser.add_argument(
        "--flow_permutation",
        type=str,
        default="invconv",
        choices=["invconv", "shuffle", "reverse"],
        help="Type of flow permutation",
    )

    parser.add_argument(
        "--flow_coupling",
        type=str,
        default="affine",
        choices=["additive", "affine"],
        help="Type of flow coupling",
    )

    parser.add_argument(
        "--no_LU_decomposed",
        action="store_false",
        dest="LU_decomposed",
        help="Train with LU decomposed 1x1 convs",
    )

    parser.add_argument(
        "--no_learn_top",
        action="store_false",
        help="Do not train top layer (prior)",
        dest="learn_top",
    )

    parser.add_argument(
        "--y_condition", action="store_true", help="Train using class condition"
    )

    parser.add_argument(
        "--extra_condition", action="store_true", help="Extra conditioning"
    )

    parser.add_argument(
        "--sp_condition", action="store_true", help="split prior conditioning"
    )

    parser.add_argument(
        "--d_condition", action="store_true", help="Train using domain conditioning"
    )

    parser.add_argument(
        "--yd_condition", action="store_true", help="Train using label and domain conditioning"
    )

    parser.add_argument(
        "--y_weight", type=float, default=0.01, help="Weight for class condition loss"
    )

    parser.add_argument(
        "--d_weight", type=float, default=0.01, help="Weight for class condition loss"
    )

    parser.add_argument(
        "--max_grad_clip",
        type=float,
        default=0,
        help="Max gradient value (clip above - for off)",
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0,
        help="Max norm of gradient (clip above - 0 for off)",
    )

    parser.add_argument(
        "--n_workers", type=int, default=6, help="number of data loading workers"
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size used during training"
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="batch size used during evaluation",
    )

    parser.add_argument(
        "--epochs", type=int, default=250, help="number of epochs to train for"
    )

    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    parser.add_argument(
        "--warmup",
        type=float,
        default=5,
        help="Use this number of epochs to warmup learning rate linearly from zero to learning rate",  # noqa
    )

    parser.add_argument(
        "--n_init_batches",
        type=int,
        default=8,
        help="Number of batches to use for Act Norm initialisation",
    )

    parser.add_argument(
        "--no_cuda", action="store_false", dest="cuda", help="Disables cuda"
    )

    parser.add_argument(
        "--output_dir",
        default="output/",
        help="Directory to output logs and model checkpoints",
    )

    parser.add_argument(
        "--fresh", action="store_true", help="Remove output directory before starting"
    )

    parser.add_argument(
        "--saved_model",
        default="",
        help="Path to model to load for continuing training",
    )
    
    parser.add_argument(
        "--missing",
        type=str,
        default=" ",
        help="missing domain",
    )

    parser.add_argument(
        "--saved_optimizer",
        default="",
        help="Path to optimizer to load for continuing training",
    )

    parser.add_argument("--seed", type=int, default=0, help="manual seed")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir,str(args.seed))
    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        if args.fresh:
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        if (not os.path.isdir(args.output_dir)) or (
            len(os.listdir(args.output_dir)) > 0
        ):
            raise FileExistsError(
                "Please provide a path to a non-existing or empty directory. Alternatively, pass the --fresh flag."  # noqa
            )

    kwargs = vars(args)
    del kwargs["fresh"]

    with open(os.path.join(args.output_dir, "hparams.json"), "w") as fp:
        json.dump(kwargs, fp, sort_keys=True, indent=4)

    main(**kwargs)
