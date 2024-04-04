# Nanofold
This project aims to build a machine learning model capable of predicting protein structure from
a given residue sequence. It uses the [Alphafold](https://www.nature.com/articles/s41586-021-03819-2)
paper as a basis, modifying details where necessary to allow for training and inference on a single
machine on a mid tier GPU.

## Setup
### Download Required Data
The download script uses `aria2c` under the hood which can be installed by running `sudo apt install aria2`.

Download and unzip `mmCIF` files that were deposited before a specified date with the following invocation:
```bash
./scripts/download_pdb.sh ~/data/pdb 2005-01-01
gzip -d ~/data/pdb/*.cif.gz
```

Download and unzip small BFD (17GB) with
```bash
wget https://storage.googleapis.com/alphafold-databases/reduced_dbs/bfd-first_non_consensus_sequences.fasta.gz
gzip -d bfd-first_non_consensus_sequences.fasta.gz
```

### Docker
Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for GPU support within containers.

Build docker images with
```bash
docker-compose build
```

Run tests with
```bash
docker run -it --gpus all --rm train pytest tests/training
docker run -it --rm data_processing pytest tests/data_processing
```

### Training
Preprocess training data by parsing downloaded mmCIF files and building multiple sequence alignments:
```bash
docker-compose run --rm data_processing python preprocess.py -m /data/pdb/ -o /preprocess/ --small_bfd /data/bfd-first_non_consensus_sequences.fasta
```

Run the training script:
```bash
docker-compose run --rm train python train.py -c config/config.ini -i /preprocess/features.arrow --mlflow
```

Run the pytorch profiler:
```bash
docker-compose run --rm train python profiler.py -c config/config.ini -i /preprocess/features.arrow
```

Refer to [this Github comment](https://github.com/pytorch/pytorch/issues/99615#issuecomment-1827386273) if the profiler is complaining with `CUPTI_ERROR_NOT_INITIALIZED`.
