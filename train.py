import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from icecream import ic
from loguru import logger as log

from test import test
from util.constants import *
from util.dataset import prepare_loader, get_num_classes
from util.model import AutoSpecNet
from util.helpers import format_significant



def train(ep,
          loader : torch.utils.data.DataLoader, 
          model, 
          criterion,
          optimizer : torch.optim.Optimizer, 
          scheduler=None):
    model.train()

    loss_meter = 0
    acc_meter = 0
    year_acc_meter = 0
    make_acc_meter = 0
    type_acc_meter = 0

    i = 0
    for data, target, year_target, make_target, type_target in (pbar:=tqdm(loader, unit=' batch', postfix=f'{len(loader.dataset)} Samples')):
        i += 1
        # Get the inputs
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        year_target = year_target.to(DEVICE)
        make_target = make_target.to(DEVICE)
        type_target = type_target.to(DEVICE)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # Make Prediction using Model
        pred, year_pred, make_pred, type_pred = model(data)

        loss_main = criterion(ic(pred), ic(target))
        loss_make = criterion(ic(make_pred), ic(make_target))
        loss_type = criterion(ic(type_pred), ic(type_target))
        loss_year = criterion(ic(year_pred), ic(year_target))
        loss = loss_main + MAKE_LOSS * loss_make + TYPE_LOSS * loss_type + YEAR_LOSS * loss_year

        loss.backward()
        optimizer.step()

        acc = pred.max(1)[1].eq(target).float().mean()
        year_acc = year_pred.max(1)[1].eq(year_target).float().mean()
        make_acc = make_pred.max(1)[1].eq(make_target).float().mean()
        type_acc = type_pred.max(1)[1].eq(type_target).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        year_acc_meter += year_acc.item()
        make_acc_meter += make_acc.item()
        type_acc_meter += type_acc.item()

        pbar.set_description(f'[TRAIN] Epoch {ep:03d}/{NUM_EPOCHS:03d} '
            f'| Loss: {format_significant(loss_meter / i)} '
            f'| Acc: {format_significant(acc_meter / i)} '
            f'| Year: {format_significant(year_acc_meter / i)} '
            f'| Make: {format_significant(make_acc_meter / i)} '
            f'| Type: {format_significant(type_acc_meter / i)} '
            )

    if scheduler:
        scheduler.step()

    loss_meter /= len(loader)
    acc_meter /= len(loader)
    year_acc_meter /= len(loader)
    make_acc_meter /= len(loader)
    type_acc_meter /= len(loader)

    trainres = {
        'train_loss': loss_meter,
        'train_acc': acc_meter,
        'train_year_acc': year_acc_meter,
        'train_make_acc': make_acc_meter,
        'train_type_acc': type_acc_meter,
    }

    return trainres


def main():
    num_classes, num_years, num_makes, num_types = get_num_classes()
    model = AutoSpecNet(BASE_MODEL, num_classes, num_years, num_makes, num_types)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = OPTIM(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # optimizer = OPTIM(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 150], gamma=0.1)
    train_loader, test_loader = prepare_loader()

    best_acc = 0
    res = []

    for ep in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        trainres = train(ep, train_loader, model, criterion, optimizer, lr_scheduler)
        valres = test(ep, test_loader, model, criterion)
        log.info(f'Test Result: Loss: {valres["val_loss"]:.4f} Acc: {valres["val_acc"]:.4f} ({(time.time() - start_time):.2f}s)')
        trainres.update(valres)

        if best_acc < (new_acc:=valres['val_acc']):
            best_acc = new_acc
            torch.save(model.state_dict(), MODEL_PATH)
            log.debug(f'Model saved to {MODEL_PATH}')
        
        res.append(trainres)
    
    log.info(f'Best accuracy: {best_acc:.4f}')
    res = pd.DataFrame(res)
    res.to_csv('history.csv')


if __name__ == '__main__':
    ic(get_num_classes())
    # ic.disable()
    # main()