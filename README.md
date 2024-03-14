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

Download sequences in FASTA format:
1) Download [`pdb_seqres.txt`](https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt),
a file containing all PDB protein sequences in FASTA format.
2) Run `python scripts/process_pdb_seqres.py $SEQRES_FILE` to process the file.

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
Process downloaded PDB files using Spark
```bash
docker-compose run --rm data_processing python process_pdb.py -m /data/pdb/ -s /db/
```
