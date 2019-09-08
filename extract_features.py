import os
import librosa
import argparse
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def compute_melspec(filename, outdir):

    wav = librosa.load(filename, sr=22050)[0]
    melspec = librosa.feature.melspectrogram(
        wav,
        sr=22050,
        n_fft=128*20,
        hop_length=347*2,
        n_mels=128,
        fmin=20,
        fmax=22050 // 2)
    logmel = librosa.core.power_to_db(melspec)
    np.save(outdir + os.path.basename(filename).split('.')[0] + '.npy', logmel)


def extract_features(out_dir, inp_txt, num_threads):
    # Create a folder to store the extracted features
    feat_folder = os.path.normpath(out_dir) + '/' + 'melspec/'
    os.makedirs(feat_folder, exist_ok=True)

    # Reading the input file
    with open(inp_txt, 'r') as f:
        lines = f.readlines()

    files = [line.rstrip() for line in lines]

    _ = Parallel(n_jobs=num_threads)(
        delayed(lambda x: compute_melspec(x, feat_folder))(x)
        for x in tqdm(files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scratch', help='Path to scratch folder')
    parser.add_argument('-i', '--input_file', help='ASCII text file with all the audio file paths')
    parser.add_argument('-n', '--num_threads', type=int, default=4, help='Num of threads to use')

    args = parser.parse_args()
    extract_features(args.scratch, args.input_file, args.num_threads)
    print('Extracting features completed')
