# Differences to Alphafold

The goal of Nanofold is not to replicate Alphafold 1 to 1 - primarily the project was started as a learning project. Moreover, the goal was to adapt the Alphafold architecture to a much smaller scale, in particular to be trainable on a single RTX3060 6GB GPU. Nanofold therefore does not attempt to achieve the accuracy of Alphafold and as such, a number of details are different to the original models. This page lists some of the key differences.

## General Differences
* Nanofold only supports protein monomers. Alphafold 3 supports protein multimers, DNA, RNA, ligands, and more.
* In order to simplify the problem space, Nanofold only attempts to predict the "backbone atoms" of each amino acid - i.e. the co-ordinates of the nitrogen, carbon, and alpha carbon atoms.

## Data Processing Differences
* Alphafold searches a large number of genetic databases to construct the MSA. Nanofold limits searches to only small BFD and Uniclust30 databases.
* Due to the length of time required to build MSAs, faster but less accurate genetic tooling is used (HHblits vs HHsearch).

## Training Differences
* The Nanofold model (listed in `config/config.json`) contains ~50 million parameters, 10x smaller than Alphafold.
* Training is done on a significantly smaller residue crop size (96 vs 384+) and MSA size (1024 sequences vs 16384).
* Alphafold uses distillation datasets (i.e. trains on predictions from Alphafold 2/3). Nanofold only trains on the original PDB dataset.
* Alphafold 3 contains many auxiliary heads, e.g. it predicts the aligned error. Many of these auxiliary heads are left out in the Nanofold architecture, saving training time on computing the Diffusion Module rollout required by these heads.
