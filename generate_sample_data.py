import os, argparse, random
from shutil import copyfile


def generate_sample_data(data_folder, input_wav):

    data_folder = os.path.normpath(data_folder)
    feat_file = open(f'{data_folder}/features_extraction.txt', 'w')
    train_file = open(f'{data_folder}/train.txt', 'w')
    test_file = open(f'{data_folder}/test.txt', 'w')
    raw_dir = f'{data_folder}/raw'
    os.makedirs(raw_dir, exist_ok=True)

    labels = ['label1', 'label2', 'label3', 'label4', 'label5']
    for i in range(1438):
        fname = os.path.abspath(f'{raw_dir}/sample{i}.wav')
        feat_file.write(f'{fname}\n')
        copyfile(input_wav, fname)
        if i <1000:
            label = random.choice(labels)
            train_file.write(f'{fname}\t{label}\n')
        else:
            test_file.write(f'{fname}\n')
            
    feat_file.close()
    train_file.close()
    test_file.close()

if __name__=="__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_folder', help='Path to data folder')
    parser.add_argument('-i', '--input_file', help='sample wav file for testing')

    args = parser.parse_args()
    generate_sample_data(args.data_folder, args.input_file)
    print('Generating sample data completed')