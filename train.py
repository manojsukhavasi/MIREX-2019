import pickle, os
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import get_train_val_data, AudioDataset
from model import Task5Model


def train(out_dir, inp_txt, num_threads):
    
    melspec_dir = os.path.normpath(out_dir)+ '/melspec'
    #create a model directory
    model_dir = os.path.normpath(out_dir) + '/' + 'model'
    os.makedirs(model_dir, exist_ok=True)

    train_fnames, val_fnames, train_labels, val_labels = get_train_val_data(inp_txt)

    train_dataset = AudioDataset(train_fnames, train_labels, melspec_dir)
    val_dataset = AudioDataset(val_fnames, val_labels, melspec_dir, train=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    model = Task5Model(5)
    print(model)

    # Define optimizer, scheduler and loss criteria
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    cuda=False
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
        for inputs,labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                model = model.train()
                outputs = model(inputs)
                # calculate loss for each set of annotations
                loss = criterion(outputs, labels)
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

        this_epoch_train_loss /= len(train_loader)
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

        print(f'Epoch: {i+1}\ttrain_loss: {this_epoch_train_loss}\tval_loss: this_epoch_valid_loss')

        scheduler.step(this_epoch_valid_loss)


if __name__=="__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('-s', '--scratch', help='Path to scratch folder')
    parser.add_argument('-i', '--input_file', help='ASCII text file with train labels')
    parser.add_argument('-n', '--num_threads', type=int, default=4, help='Num of threads to use')

    args = parser.parse_args()
    train(args.scratch, args.input_file, args.num_threads)
    print('Training completed')