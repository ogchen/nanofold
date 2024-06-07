# Nanofold
This project implements a protein structure prediction machine learning model using the [Alphafold 2](https://www.nature.com/articles/s41586-021-03819-2) and [Alphafold 3](https://www.nature.com/articles/s41586-024-07487-w) papers as a basis, designed to be trainable on a single mid tier GPU.

- [Nanofold](#nanofold)
  - [Features](#features)
  - [Implementation Details](#implementation-details)
    - [Data Processing Pipeline](#data-processing-pipeline)
    - [Training Pipeline](#training-pipeline)
  - [Setup](#setup)
    - [Download Required Data](#download-required-data)
    - [Docker](#docker)
  - [Training](#training)
  - [Profiling](#profiling)
  - [Running Unit Tests](#running-unit-tests)

## Documentation
See [ogchen.github.io/nanofold](https://ogchen.github.io/nanofold/) for project documentation and [this blog post](https://medium.com/@ogchen/the-worst-method-for-learning-ml-reproducing-alphafold-8e9badfacce9) for a project write up.

## Features
* Leverages the `Alphafold 3` architecture which is significantly more efficient than the equivalent `Alphafold 2` modules. Restricts the problem space to monomer protein chains to reduce training data required.
* Reduces GPU memory usage with [gradient checkpointing](https://pytorch.org/docs/stable/checkpoint.html).
* Training is done using [bfloat16](https://pytorch.org/docs/stable/amp.html), further reducing GPU memory footprint.
* Uses [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html) for JIT compilation for training speedup.
* Stores input features in [Apache Arrow](https://arrow.apache.org/docs/index.html)'s IPC format to handle datasets larger than available RAM.
* Integration with [MLFlow](https://mlflow.org/) to monitor training metrics and manage model checkpoints.
* Compression of dataset using sparse matrices to save disk space.
* [Docker](https://www.docker.com/) images for training and data processing pipeline.
* CI for running python tests with [GitHub Actions](https://docs.github.com/en/actions).
* Support for [development within containers](https://code.visualstudio.com/docs/devcontainers/containers).


## Implementation Details
### Data Processing Pipeline
The data processing pipeline (entry point at `nanofold/preprocess/__main__.py`) performs the following steps:
* Parses mmCIF files from the [Protein Data Bank](https://www.rcsb.org/) for protein chain details, including the residue sequence and atom co-ordinates.
* For each protein chain, it searches the [small BFD](https://bfd.mmseqs.com/) and [Uniclust30](https://uniclust.mmseqs.com/) genetic databases for proteins with similar residue sequences. The results are combined to form the multiple sequence alignment (MSA).
* Using the MSA, we search another database (PDB70) to find "templates" - proteins that are structurally similar.
* Dumps all input features to an Arrow IPC file, ready for the training pipeline.

### Training Pipeline
Nanofold largely implements the model algorithms detailed in [Alphafold's Supplementary Information](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-07487-w/MediaObjects/41586_2024_7487_MOESM1_ESM.pdf), with a few key exceptions:
* In order to simplify the problem, Nanofold only considers protein chains in isolation. All details regarding ligands, DNA, RNA, and other small molecules, are ignored. Furthermore, there is only support for single chain proteins (monomers).
* Alphafold 3 implements additional auxiliary heads, i.e. the model is trained to predict various metrics such as the predicted local distance difference. These are ignored in Nanofold.

The relevant code can be found in `nanofold/train`.

## Setup
### Download Required Data
The following tools are required for the download script:
```bash
sudo apt install aria2 jq
```

Choose a download directory and a cut off date for PDB files. Note: adjust `DATA_DIR` in `docker/.env` if you choose a path
other than `$HOME/data`.

Download and unzip `mmCIF` files that were deposited before the cut off date with the following invocation:
```bash
export DATA_DIR=~/data
export CUTOFF_DATE=1989-01-01
mkdir -p $DATA_DIR/pdb && ./scripts/download_pdb.sh $DATA_DIR/pdb $CUTOFF_DATE
```

Download and unzip small BFD (17GB) with
```bash
wget https://storage.googleapis.com/alphafold-databases/reduced_dbs/bfd-first_non_consensus_sequences.fasta.gz -P $DATA_DIR
gzip -d $DATA_DIR/bfd-first_non_consensus_sequences.fasta.gz
```

Download and unzip Uniclust30 with
```bash
aria2c https://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/uniclust30_2016_03.tgz -d $DATA_DIR
tar -xf $DATA_DIR/uniclust30_2016_03.tgz -C $DATA_DIR
```

Download and unzip PDB70 (56GB) for template search with
```bash
wget https://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz -P $DATA_DIR
mkdir -p $DATA_DIR/pdb70 && tar -xf $DATA_DIR/pdb70_from_mmcif_200401.tar.gz -C $DATA_DIR/pdb70
```

### Docker
Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for GPU support within containers.

Build the docker images with
```bash
docker-compose -f docker/docker-compose.preprocess.yml build
docker-compose -f docker/docker-compose.train.yml build
```

## Training
Run the preprocessing script with
```bash
docker-compose -f docker/docker-compose.preprocess.yml run --rm preprocess python -m nanofold.preprocess -m /data/pdb/ -c /preprocess/ -o /preprocess/features.arrow --small_bfd /data/bfd-first_non_consensus_sequences.fasta --pdb70 /data/pdb70/pdb70 --uniclust30 /data/uniclust30_2016_03/uniclust30_2016_03
```
Note: this step takes a long time depending on the number of input proteins. One potential future adaptation would be to test out the speed of `MMseqs2`, or to use precomputed MSAs.

Run the training script for `N` epochs:
```bash
docker-compose -f docker/docker-compose.train.yml run --rm train python -m nanofold.train -c config/config.json -i /preprocess/features.arrow --mlflow --max-epoch $N
```
The training script will also spin up an MLFlow server. The dashboard can be accessed at [localhost:8000](http://localhost:8000/) to monitor training metrics.

To resume training from an MLFlow checkpoint, identify the corresponding `$RUNID` and run:
```bash
docker-compose -f docker/docker-compose.train.yml run --rm train python -m nanofold.train -r $RUNID -i /preprocess/features.arrow --mlflow --max-epoch $N
```

## Profiling
Run the pytorch profiler:
```bash
docker-compose -f docker/docker-compose.train.yml run --rm -v $DATA_DIR:/data train python -m nanofold.profile -c config/config.json -i /preprocess/features.arrow --mode time --mode memory
```
The profiler spits out a `trace.json` and `snapshot.pickle` file in the mounted `/data/` volume.
Load `trace.json` into [chrome://tracing](chrome://tracing/), and `snapshot.pickle` into [pytorch.org/memory_viz](https://pytorch.org/memory_viz).

Refer to [this Github comment](https://github.com/pytorch/pytorch/issues/99615#issuecomment-1827386273) if the profiler is complaining with `CUPTI_ERROR_NOT_INITIALIZED`.

## Running Unit Tests
Run tests with
```bash
docker run --rm --gpus all train pytest tests/train
docker run --rm preprocess pytest tests/preprocess
```
