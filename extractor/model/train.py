from .model import build_training_model
from .dataloader import build_dataloader
from .loss import TripletLoss
import torch.nn.functional as F
import torch
import time
import os
import pickle
import utils.configs as cfg
import datetime


def train():
    dataloader, catid_to_classid = build_dataloader(cfg)
    model = build_training_model(cfg)

    criterion = TripletLoss(margin=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.SCHEDULER.STEP, gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    
    dir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    num_epochs = cfg.TRAIN.EPOCHS
    for epoch in range(num_epochs):
        
        run_epoch(model, criterion, dataloader, epoch, scheduler, optimizer)
        scheduler.step()
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, f'{dir_name}/{epoch}.pth')


def run_epoch(model, criterion, dataloader, epoch, scheduler, optimizer):
    print_freq = 10
    htri_losses = AverageMeter()
    xent_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accs = AverageMeter()

    end = time.time()
    for batch_idx, (imgs, pids, camid, _) in enumerate(dataloader):
        data_time.update(time.time() - end)
        imgs, pids, camid = imgs.cuda(), pids.cuda(), camid.cuda()

        outputs, clas = model(imgs)
        # print(pids)
        htri_loss = criterion(outputs, pids)
        # clas = model[1(outputs)
        # xent_loss = F.cross_entropy(clas, camid)
        loss = htri_loss# + xent_loss
        # loss = xent_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        htri_losses.update(htri_loss.item(), pids.size(0))
        # xent_losses.update(xent_loss.item(), pids.size(0))
        accs.update(accuracy(clas, camid)[0])
        if (batch_idx + 1) % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Xent {xent.val:.4f} ({xent.avg:.4f})\t'
                  'Htri {htri.val:.4f} ({htri.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                epoch + 1, batch_idx + 1, len(dataloader),
                batch_time=batch_time,
                data_time=data_time,
                xent=xent_losses,
                htri=htri_losses,
                acc=accs
            ))

        end = time.time()


def train_pca():
    pass
