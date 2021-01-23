import torch
import torch.nn as nn
import torch.optim as optim

import torchutils
from torchutils.metrics import EstimatedTimeArrival
from torchutils.distributed import torchsave
from torchutils.common import compute_nparam, compute_flops
from torchutils.common import save_checkpoint
from torchutils.common import set_cudnn_auto_tune

from codebase.config import args
from codebase.scheduler import build_scheduler
from codebase.optimizer import build_optimizer
from codebase.criterion import CRITERION
from codebase.dataset import build_imagenet_loader
from codebase.models import build_model
from codebase.engine import train, evaluate


if __name__ == "__main__":
    torchutils.logger.info(args)
    set_cudnn_auto_tune()

    # distributed init

    model: nn.Module = build_model(args)
    torchutils.logger.info(f"Model details: n_params={compute_nparam(model)/1e6:.2f}M, "
                           f"flops={compute_flops(model,(1,3,args.image_size, args.image_size))/1e6:.2f}M.")

    train_loader, val_loader = build_imagenet_loader(args)

    criterion = CRITERION.build_from(args)
    optimizer: optim.Optimizer = build_optimizer(model.parameters(), args)
    scheduler = build_scheduler(optimizer, args)

    epoch = 0
    max_epochs = args.max_epochs
    ETA = EstimatedTimeArrival(max_epochs)

    top1_accuracy, top5_accuracy = 0.0, 0.0

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        start_epoch = ckpt["epoch"]
        best_acc1 = ckpt["best_acc1"]
        best_acc5 = ckpt["best_acc5"]
        best_epoch = ckpt["best_epoch"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        torchutils.logger(f"Load chckpoint from {args.resume}, epoch={epoch}, best_acc1={best_acc1*100:.2f}%.")
    else:
        start_epoch = 0
        best_epoch = 0
        best_acc1, best_acc5 = 0.0, 0.0

    if torchutils.distributed.is_dist_avail_and_init():
        torch.cuda.set_device(utils.distributed.local_rank())
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[utils.distributed.local_rank()])

    for epoch in range(start_epoch+1, max_epochs+1):
        for func, loader in zip((train, evaluate), (train_loader, val_loader)):
            name = func.__name__.upper()
            loss, (top1_accuracy, top5_accuracy) = func(epoch, model, loader, criterion,
                                                        optimizer, scheduler, args.amp, args.report_freq)
            torchutils.summary_writer.add_scalar(f"{name}/loss", loss, epoch)
            torchutils.summary_writer.add_scalar(f"{name}/acc_1", top1_accuracy, epoch)
            torchutils.summary_writer.add_scalar(f"{name}/acc_5", top5_accuracy, epoch)
            torchutils.summary_writer.add_scalar(f"{name}/lr", optimizer.param_groups[0]['lr'], epoch)
            torchutils.logger.info(", ".join([
                f"{name} Complete",
                f"epoch={epoch:04d}",
                f"loss={loss:.4f}",
                f"top1-accuracy={top1_accuracy*100:.2f}%",
                f"top5-accuracy={top5_accuracy*100:.2f}%",
            ]))
        ETA.step()
        torchutils.logger.info(", ".join([
            f"Epoch Complete",
            f"epoch={epoch:03d}",
            f"best acc={best_acc1*100:.2f}%/{best_acc5*100:.2f}%(epoch={best_epoch:03d})",
            # f"lr={optimizer.param_groups[0]['lr']:.6f}",
            f"eta={ETA.remaining_time}",
            f"arrival={ETA.arrival_time}",
            f"cost={ETA.cost_time}",
        ]))
        if best_acc1 < top1_accuracy:
            best_acc1 = top1_accuracy
            best_acc5 = top5_accuracy
            best_epoch = epoch
        save_checkpoint(utils.output_directory, epoch, model, optimizer,
                        best_acc1, best_acc5, best_epoch)
