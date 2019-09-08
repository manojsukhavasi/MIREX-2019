import os
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import get_train_val_data, AudioDataset
from model import MirexModel
from config import CONFIG


# from https://github.com/tugstugi/pytorch-speech-commands/blob/master/mixup.py
def mixup_cross_entropy_loss(inp, target, size_average=True):
    """Origin: https://github.com/moskomule/mixup.pytorch
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the inp
    """
    assert inp.size() == target.size()
    inp = torch.log(torch.nn.functional.softmax(inp, dim=1).clamp(1e-5, 1))
    loss = -torch.sum(inp * target)
    return loss / inp.size()[0] if size_average else loss


def train(out_dir, inp_txt, num_threads, task, batch_size=4):

    torch.set_num_threads(num_threads)
    print('Number of threads: ', torch.get_num_threads())

    melspec_dir = os.path.normpath(out_dir) + '/melspec'

    # create a model directory
    model_dir = os.path.normpath(out_dir) + '/' + 'model'
    os.makedirs(model_dir, exist_ok=True)

    train_fnames, val_fnames, train_labels, val_labels = get_train_val_data(inp_txt)

    train_dataset = AudioDataset(train_fnames, train_labels, melspec_dir)
    val_dataset = AudioDataset(val_fnames, val_labels, melspec_dir, train=False)

    train_loader_1 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_2 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    num_classes = CONFIG[task]['num_classes']

    model = MirexModel(num_classes)

    # Define optimizer, scheduler and loss criteria
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    cuda = False
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Device: ', device)
    model = model.to(device)

    epochs = 100
    train_loss_hist = []
    valid_loss_hist = []
    lowest_val_loss = np.inf
    epochs_without_new_lowest = 0

    for i in range(epochs):

        this_epoch_train_loss = 0
        for i1, i2 in zip(train_loader_1, train_loader_2):

            # mixup---------
            x1, y1 = i1
            x2, y2 = i2

            alpha = 1
            mixup_vals = np.random.beta(alpha, alpha, i1[0].shape[0])

            mvals = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1, 1, 1))
            inputs = (mvals * x1) + ((1 - mvals) * x2)

            y1_onehot = torch.nn.functional.one_hot(y1, num_classes).float()
            y2_onehot = torch.nn.functional.one_hot(y2, num_classes).float()
            mvals = torch.Tensor(mixup_vals.reshape(mixup_vals.shape[0], 1))
            labels = (mvals * y1_onehot) + ((1 - mvals) * y2_onehot)
            # mixup ends ----------

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                model = model.train()
                outputs = model(inputs)
                loss = mixup_cross_entropy_loss(outputs, labels, False)
                loss.backward()
                optimizer.step()
                this_epoch_train_loss += loss.detach().cpu().numpy()

        this_epoch_valid_loss = 0
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                model = model.eval()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                this_epoch_valid_loss += loss.detach().cpu().numpy()

        this_epoch_train_loss /= len(train_loader_1)
        this_epoch_valid_loss /= len(val_loader)

        train_loss_hist.append(this_epoch_train_loss)
        valid_loss_hist.append(this_epoch_valid_loss)

        if this_epoch_valid_loss < lowest_val_loss:
            lowest_val_loss = this_epoch_valid_loss
            torch.save(model.state_dict(), f'{model_dir}/best_model.pth')
            epochs_without_new_lowest = 0
        else:
            epochs_without_new_lowest += 1

        if epochs_without_new_lowest >= 25:
            break

        print(f'Epoch: {i+1}\ttrain_loss: {this_epoch_train_loss}\tval_loss: {this_epoch_valid_loss}')

        scheduler.step(this_epoch_valid_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scratch', help='Path to scratch folder')
    parser.add_argument('-i', '--input_file', help='ASCII text file with train labels')
    parser.add_argument('-n', '--num_threads', type=int, default=4, help='Num of threads to use')
    parser.add_argument('-t', '--task', type=str, default='kpop_mood',
                        help='Task name, see config for choices')

    args = parser.parse_args()
    train(args.scratch, args.input_file, args.num_threads, args.task)
    print('Training completed')
