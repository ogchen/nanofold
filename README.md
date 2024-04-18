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

Download and unzip PDB70 (56GB) for template search with
```bash
wget https://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz
mkdir pdb70 && tar -xf pdb70_from_mmcif_200401.tar.gz -C pdb70
```

### Docker
Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for GPU support within containers.

Build the docker images with
```bash
docker-compose -f docker/docker-compose.process.yml build
docker-compose -f docker/docker-compose.train.yml build
```

## Training
Run the preprocessing script with
```bash
docker-compose -f docker/docker-compose.process.yml run --rm data_processing python preprocess.py -m /data/pdb/ -o /preprocess/ --small_bfd /data/bfd-first_non_consensus_sequences.fasta
```
This parses the downloaded mmCIF files to extract protein information, including the residue sequence and atom co-ordinates.
It uses the jackhmmer tool to search the provided small BFD database and build multiple sequence alignments, before clustering
and computing various features to be used in training.

Run the training script for `N` epochs:
```bash
docker-compose -f docker/docker-compose.train.yml run --rm train python train.py -c config/config.json -i /preprocess/features.arrow --mlflow --max-epoch $N
```

To resume training from an MLFlow checkpoint, identify the corresponding `$RUNID` and run:
```bash
docker-compose -f docker/docker-compose.train.yml run --rm train python train.py -r $RUNID -i /preprocess/features.arrow --mlflow --max-epoch $N
```

## Profiling
Run the pytorch profiler:
```bash
docker run --rm -v $HOME/data:/data train python profiler.py -c config/config.json -i /preprocess/features.arrow --mode time --mode memory
```
The profiler spits out a `trace.json` and `snapshot.pickle` file in the mounted `/data/` volume.
Load `trace.json` into [chrome://tracing](chrome://tracing/), and `snapshot.pickle` into [pytorch.org/memory_viz](https://pytorch.org/memory_viz).

Refer to [this Github comment](https://github.com/pytorch/pytorch/issues/99615#issuecomment-1827386273) if the profiler is complaining with `CUPTI_ERROR_NOT_INITIALIZED`.

## Unit Tests
Run tests with
```bash
docker run --rm --gpus all train pytest tests/training
docker run --rm data_processing pytest tests/data_processing
```
