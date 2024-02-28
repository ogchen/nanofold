# Nanofold
This project aims to build a machine learning model capable of predicting protein structure from
a given residue sequence. It uses the [Alphafold](https://www.nature.com/articles/s41586-021-03819-2)
paper as a basis, modifying details where necessary to allow for training and inference on a single
machine on a mid tier GPU.

## Setup
### Download Required Data
Download the PDB structures used as the primary training inputs:
1) Use RCSB's [advanced search](https://www.rcsb.org/search/advanced) to filter protein structures
deposited before 01/01/2000, limiting the number of inputs sequences to 10959.
2) Download the list of IDs resulting from the search above, and use the RCSB
[batch script](https://www.rcsb.org/docs/programmatic-access/batch-downloads-with-shell-script) to
download all structures in the `mmCIF` format.
3) Unzip all files with `gzip -d *.cif.gz`.

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
```bash
docker build -t nanofold .
docker run -it nanofold /bin/bash
```