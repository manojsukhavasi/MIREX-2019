# Instructions to run the algorithm for mirex 2019 tasks

## Audio Classical Composer Identification

### Environment and setup

- Download `mirex.tar.gz`. This is a conda package which has all the libraries required.
- Create a folder for environment setup `mkdir -p mirex2019_sm`
- Unpack the package into the folder created `tar -xzf mirex.tar.gz -C mirex2019_sm`
- Activate the conda environment in `bash` shell: `source mirex2019_sm/bin/activate`
- Run the command `conda-unpack`
- unzip the code directory

### Feature extraction
- Disk space requirements ~ 3GB
- Time Taken ~ 10min
- `python extract_features.py --scratch {path_to_scratch_folder} --input_file {feature_extraction_list_file} --num_threads 4`

### Training
- Disk space requirements ~ None
- Time Taken ~ 10-15Hrs
- `python train.py --scratch {path_to_scratch_folder} --input_file {train_list_file} --num_threads 4`
- If you get a Memory Error, please use `--batch_size` parameter to decrease the batch size to 16/8/4

### Classification
- Disk space requirements ~ None
- Time Taken ~ 30min
- `python classify.py --scratch {path_to_scratch_folder} --input_file {test_list_file} --out_file {output_list_file} --num_threads 4`


## Audio US Pop Music Genre Classification
### Environment and setup

- Download `mirex.tar.gz`. This is a conda package which has all the libraries required.
- Create a folder for environment setup `mkdir -p mirex2019_sm`
- Unpack the package in to the folder created `tar -xzf mirex.tar.gz -C mirex2019_sm`
- Activate the conda enviroment `source mirex2019_sm/bin/activate`
- `conda-unpack`
- unzip the code directory

### Feature extraction
- Disk space requirements ~ 10 GB
- Time Taken ~ 30min
- `python extract_features.py --scratch {path_to_scratch_folder} --input_file {feature_extraction_list_file} --num_threads 4`

### Training
- Disk space requirements ~ None
- Time Taken ~ 25-40Hrs
- `python train.py --scratch {path_to_scratch_folder} --input_file {train_list_file} --num_threads 4`
- If you get a Memory Error, please use `--batch_size` parameter to decrease the batch size to 16/8/4

### Classification
- Disk space requirements ~ None
- Time Taken ~ 90min
- `python classify.py --scratch {path_to_scratch_folder} --input_file {test_list_file} --out_file {output_list_file} --num_threads 4`

## Audio Latin Music Genre Classification
### Environment and setup

- Download `mirex.tar.gz`. This is a conda package which has all the libraries required.
- Create a folder for environment setup `mkdir -p mirex2019_sm`
- Unpack the package in to the folder created `tar -xzf mirex.tar.gz -C mirex2019_sm`
- Activate the conda enviroment `source mirex2019_sm/bin/activate`
- `conda-unpack`
- unzip the code directory

### Feature extraction
- Disk space requirements ~ 4GB
- Time Taken ~ 10min
- `python extract_features.py --scratch {path_to_scratch_folder} --input_file {feature_extraction_list_file} --num_threads 4`

### Training
- Disk space requirements ~ None
- Time Taken ~ 10-15Hrs
- `python train.py --scratch {path_to_scratch_folder} --input_file {train_list_file} --num_threads 4`
- If you get a Memory Error, please use `--batch_size` parameter to decrease the batch size to 16/8/4

### Classification
- Disk space requirements ~ None
- Time Taken ~ 30min
- `python classify.py --scratch {path_to_scratch_folder} --input_file {test_list_file} --out_file {output_list_file} --num_threads 4`

## Audio Music Mood Classification
### Environment and setup

- Download `mirex.tar.gz`. This is a conda package which has all the libraries required.
- Create a folder for environment setup `mkdir -p mirex2019_sm`
- Unpack the package in to the folder created `tar -xzf mirex.tar.gz -C mirex2019_sm`
- Activate the conda enviroment `source mirex2019_sm/bin/activate`
- `conda-unpack`
- unzip the code directory

### Feature extraction
- Disk space requirements ~ 1GB
- Time Taken ~ 5min
- `python extract_features.py --scratch {path_to_scratch_folder} --input_file {feature_extraction_list_file} --num_threads 4`

### Training
- Disk space requirements ~ None
- Time Taken ~ 3-5Hrs
- `python train.py --scratch {path_to_scratch_folder} --input_file {train_list_file} --num_threads 4`
- If you get a Memory Error, please use `--batch_size` parameter to decrease the batch size to 16/8/4

### Classification
- Disk space requirements ~ None
- Time Taken ~ 10min
- `python classify.py --scratch {path_to_scratch_folder} --input_file {test_list_file} --out_file {output_list_file} --num_threads 4`

## Audio K-POP Mood Classification
### Environment and setup

- Download `mirex.tar.gz`. This is a conda package which has all the libraries required.
- Create a folder for environment setup `mkdir -p mirex2019_sm`
- Unpack the package in to the folder created `tar -xzf mirex.tar.gz -C mirex2019_sm`
- Activate the conda enviroment `source mirex2019_sm/bin/activate`
- `conda-unpack`
- unzip the code directory

### Feature extraction
- Disk space requirements ~ 2GB
- Time Taken ~ 5min
- `python extract_features.py --scratch {path_to_scratch_folder} --input_file {feature_extraction_list_file} --num_threads 4`

### Training
- Disk space requirements ~ None
- Time Taken ~ 5-8Hrs
- `python train.py --scratch {path_to_scratch_folder} --input_file {train_list_file} --num_threads 4`
- If you get a Memory Error, please use `--batch_size` parameter to decrease the batch size to 16/8/4

### Classification
- Disk space requirements ~ None
- Time Taken ~ 15min
- `python classify.py --scratch {path_to_scratch_folder} --input_file {test_list_file} --out_file {output_list_file} --num_threads 4`

## Audio K-POP Genre Classification
### Environment and setup

- Download `mirex.tar.gz`. This is a conda package which has all the libraries required.
- Create a folder for environment setup `mkdir -p mirex2019_sm`
- Unpack the package in to the folder created `tar -xzf mirex.tar.gz -C mirex2019_sm`
- Activate the conda enviroment `source mirex2019_sm/bin/activate`
- `conda-unpack`
- unzip the code directory

### Feature extraction
- Disk space requirements ~ 2GB
- Time Taken ~ 10min
- `python extract_features.py --scratch {path_to_scratch_folder} --input_file {feature_extraction_list_file} --num_threads 4`

### Training
- Disk space requirements ~ None
- Time Taken ~ 7-10Hrs
- `python train.py --scratch {path_to_scratch_folder} --input_file {train_list_file} --num_threads 4`
- If you get a Memory Error, please use `--batch_size` parameter to decrease the batch size to 16/8/4

### Classification
- Disk space requirements ~ None
- Time Taken ~ 20min
- `python classify.py --scratch {path_to_scratch_folder} --input_file {test_list_file} --out_file {output_list_file} --num_threads 4`
