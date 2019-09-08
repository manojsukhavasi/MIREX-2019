import os
import argparse
import pickle

import torch
from torch.utils.data import DataLoader

from dataset import get_test_data, TestDataset
from model import MirexModel
from config import CONFIG


def classify(out_dir, inp_txt, out_file, num_threads, task, batch_size=4):
    melspec_dir = os.path.normpath(out_dir) + '/melspec'
    model_dir = os.path.normpath(out_dir) + '/' + 'model'
    best_model_path = model_dir + '/best_model.pth'
    mean_std_path = os.path.normpath(out_dir) + '/mean_std.pkl'
    labels_path = os.path.normpath(out_dir) + '/label_ids.pkl'

    with open(mean_std_path, 'rb') as f:
        mean, std = pickle.load(f)

    test_fnames = get_test_data(inp_txt)
    test_dataset = TestDataset(test_fnames, melspec_dir, mean, std)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    cuda = False
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Device: ', device)

    num_classes = CONFIG[task]['num_classes']
    model = MirexModel(num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(best_model_path))  # Loading the best model

    test_preds = []

    for inputs in test_loader:
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            model = model.eval()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
        test_preds.extend(list(predicted.numpy()))

    with open(labels_path, 'rb') as f:
        ref_labels_dict = pickle.load(f)
    ids_to_labels = {i: x for x, i in ref_labels_dict.items()}

    with open(out_file, 'w') as f:
        for i in range(len(test_fnames)):
            this_file = test_fnames[i]
            this_pred = ids_to_labels[test_preds[i]]
            f.write(f'{this_file}\t{this_pred}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scratch', help='Path to scratch folder')
    parser.add_argument('-i', '--input_file', help='ASCII text file with train labels')
    parser.add_argument('-o', '--out_file', help='ASCII text file with train labels')
    parser.add_argument('-n', '--num_threads', type=int, default=4, help='Num of threads to use')
    parser.add_argument('-t', '--task', type=str, default='kpop_mood',
                        help='Task name, see config for choices')

    args = parser.parse_args()
    classify(args.scratch, args.input_file, args.out_file, args.num_threads, args.task)
    print('Classification completed')
