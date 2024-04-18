import torch
from tqdm import tqdm

from util.dataset import prepare_loader, get_num_classes
from util.constants import *
from util.model import AutoSpecNet
from util.helpers import format_significant


def test(ep, loader: torch.utils.data.DataLoader, model, criterion):
    model.eval()

    loss_meter = 0
    acc_meter = 0
    year_acc_meter = 0
    make_acc_meter = 0
    type_acc_meter = 0
    runcount = 0

    i = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data, target, year_target, make_target, type_target in (pbar:=tqdm(loader, unit=' batch', postfix=f'{len(loader.dataset)} Samples')):
            i += 1
            # Get the test inputs
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            year_target = year_target.to(DEVICE)
            make_target = make_target.to(DEVICE)
            type_target = type_target.to(DEVICE)

            # Make Prediction using Model
            pred, year_pred, make_pred, type_pred = model(data)

            loss_main = criterion(pred, target)
            loss_make = criterion(make_pred, make_target)
            loss_type = criterion(type_pred, type_target)
            loss_year = criterion(year_pred, year_target)
            loss = loss_main + MAKE_LOSS * loss_make + TYPE_LOSS * loss_type + YEAR_LOSS * loss_year

            acc = pred.max(1)[1].eq(target).float().sum()
            year_acc = year_pred.max(1)[1].eq(year_target).float().sum()
            make_acc = make_pred.max(1)[1].eq(make_target).float().sum()
            type_acc = type_pred.max(1)[1].eq(type_target).float().sum()

            loss_meter += loss.item() * data.size(0)
            acc_meter += acc.item()
            year_acc_meter += year_acc.item()
            make_acc_meter += make_acc.item()
            type_acc_meter += type_acc.item()

            runcount += data.size(0)

            pbar.set_description(f'[TEST ] Epoch {ep:03d}/{NUM_EPOCHS:03d} '
                f'| Loss: {format_significant(loss_meter / runcount)} '
                f'| Acc: {format_significant(acc_meter / runcount)} '
                f'| Year: {format_significant(year_acc_meter / runcount)} '
                f'| Make: {format_significant(make_acc_meter / runcount)} '
                f'| Type: {format_significant(type_acc_meter / runcount)} '
                )

        loss_meter /= runcount
        acc_meter /= runcount
        year_acc_meter /= runcount
        make_acc_meter /= runcount
        type_acc_meter /= runcount

    valres = {
        'val_loss': loss_meter,
        'val_acc': acc_meter,
        'train_year_acc': year_acc_meter,
        'val_make_acc': make_acc_meter,
        'val_type_acc': type_acc_meter,
    }

    return valres


def load_weight(model, path, device):
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)


def main():

    num_classes, num_years, num_makes, num_types = get_num_classes()
    model = AutoSpecNet(BASE_MODEL, num_classes, num_years, num_makes, num_types)
    load_weight(model, MODEL_PATH, DEVICE)
    model = model.to(DEVICE)

    train_loader, test_loader = prepare_loader()
    test(test_loader, model)

