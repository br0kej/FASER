# FASER: Binary Code Similarity Search through the use of Intermediate Representations

This repository contains the source code used to generate the results presented in [FASER: Binary Code Similarity
Search through the use of Intermediate Representations](https://arxiv.org/abs/2310.03605).

# Abstract
>Being able to identify functions of interest in cross-architecture software is useful whether you are analysing for malware, securing the software supply chain or conducting vulnerability research. Cross-Architecture Binary Code Similarity Search has been explored in numerous studies and has used a wide range of different data sources to achieve its goals. The data sources typically used draw on common structures derived from binaries such as function control flow graphs or binary level call graphs, the output of the disassembly process or the outputs of a dynamic analysis approach. One data source which has received less attention is binary intermediate representations. Binary Intermediate representations possess two interesting properties: they are cross architecture by their very nature and encode the semantics of a function explicitly to support downstream usage. Within this paper we propose Function as a String Encoded Representation (FASER) which combines long document transformers with the use of intermediate representations to create a model capable of cross architecture function search without the need for manual feature engineering, pre-training or a dynamic analysis step. We compare our approach against a series of baseline approaches for two tasks; A general function search task and a targeted vulnerability search task. Our approach demonstrates strong performance across both tasks, performing better than all baseline approaches.
# Setup Steps

## 1. Clone Repository

```
git clone https://github.com/br0kej/FASER
```

## 2. Setup `python` environment
The code has been developed using `python 3.10.2` but should work on machines with at least `python 3.8`. In order to
install the dependencies execute the following commands:
```
cd FASER
python -m venv venv
source venv/bin/activate
python -m pip install .
```
## 3. Download model artefacts

Both the register normalised and non-register normalised models can be downloaded from
[Google Drive - Here](https://drive.google.com/drive/folders/1EWUAo07_hg7u3rAOVl14xiqkaYCaIK2p?usp=drive_link). Once downloaded,
place these into the `FASER/model_artefacts` directory.

## 4. Download the general function search data and the vulnerability search data

Similarly to the models, the data can be downloaded from [Google Drive - Here](https://drive.google.com/drive/folders/1EWUAo07_hg7u3rAOVl14xiqkaYCaIK2p?usp=drive_link).
This can be placed anywhere on your machine but will be used as an argument into `cli.py`.

## 5. Re-run the evaluation steps

There are two options within `cli.py` that provie a means of easily re-running the evaluation. For example, to re-run
the general function search evaluation for the `FASER-RN` model, you would execute the following command:
>You need to amend the command below to have the correct path to your downloaded data in the `-d` argument.
```
python cli.py fseval -m model_artefacts/FASER-RN.bin -t model_artefacts/4096-efs-tokeniser-pad-trunc-RN.json -d <path_to_data>/reg-normd/train/
```

Alternatively, to re-run the vulnerability search evaluation with the `FASER-RN` model, you would execute the following
command:
```
python cli.py vseval -m model_artefacts/FASER-RN.bin -t model_artefacts/4096-efs-tokeniser-pad-trunc-RN.json -d ../phd-data/Dataset-Vulnerability/processed/esil-fstrs-reg-normd/ -f faser-rn-re-run
```

The `cli.py` also has commands for training a FASER model, creating data suitable to train a tokeniser from `bin2ml` generated
data as well as some utilities such as encoding an input string to see what the post-tokenised output looks like.

# Citation

```
@misc{collyer2023faser,
      title={FASER: Binary Code Similarity Search through the use of Intermediate Representations},
      author={Josh Collyer and Tim Watson and Iain Phillips},
      year={2023},
      eprint={2310.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
