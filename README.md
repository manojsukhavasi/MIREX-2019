# Mirex 2019 K-pop mood classification

## Running commands

### generating sample data

`python generate_sample_data.py -d data/ -i data/sample.wav`

### Feature extraction

`python extract_features.py -s /home/scratch -i data/features_extraction.txt -n 4`

### Training

`python train.py -s /home/scratch -i data/train.txt -n 4`

### Classifying

`python classify.py -s /home/scratch -i data/test.txt -o test_preds.txt -n 4`

## Time taken

### Features extraction 
- 4 threads ~ 5 min
- ~ 1.5 GB memory for extracted features # maychange with parameters
