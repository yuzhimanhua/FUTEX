# Weakly Supervised Multi-Label Classification of Full-Text Scientific Papers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the datasets and source code used in our paper [Weakly Supervised Multi-Label Classification of Full-Text Scientific Papers](https://arxiv.org/abs/2306.14003).

## Links

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data](#data)
- [Running on New Datasets](#running-on-new-datasets)
- [Citation](#citation)

## Installation

For training, GPUs are required. We use one NVIDIA RTX A6000 GPU in our experiments.

### Dependency
The code is written in Python 3.6. The dependencies are summarized in the file ```requirements.txt```. You can install them like this:

```
pip3 install -r requirements.txt
```

## Quick Start
To reproduce the results in our paper, you need to first download the [**datasets**](https://drive.google.com/file/d/1lK-7cuart0h8VDpWhVy8eFGrXTmSPwt2/view?usp=drive_link). Three datasets are used in the paper: **MAG-CS**, **PubMed**, and **Art**. Once you unzip the downloaded file (i.e., ```FUTEX.zip```), you can see four folders. Three of them, ```MAGCS/```, ```PubMed/```, and ```Art/``` correspond to the three datasets, respectively. The other one, ```specter/``` is the pre-trained SPECTER model. (The pre-trained SPECTER model is from [here](https://huggingface.co/allenai/specter/tree/main). Feel free to use other pre-trained models, such as SciNCL which can be downloaded from [here](https://huggingface.co/malteos/scincl/tree/main).)

You need to put all four folders under the repository main folder ```./```. Then you need to run the following scripts.

```
./run.sh
```

P@_k_, NDCG@_k_, PSP@_k_, and PSN@_k_ scores (_k_=1,3,5) will be shown in the last several lines of the output as well as in ```./scores.txt.``` The prediction results can be found in ```./{dataset}/{dataset}_predictions_futex.json``` (e.g., ```./MAGCS/MAGCS_predictions_futex.json```).

## Data
TBD

## Running on New Datasets
TBD

## Citation
If you find this repository useful, please cite the following paper:
```
@inproceedings{zhang2023weakly,
  title={Weakly Supervised Multi-Label Classification of Full-Text Scientific Papers},
  author={Zhang, Yu and Jin, Bowen and Chen, Xiusi and Shen, Yanzhen and Zhang, Yunyi and Meng, Yu and Han, Jiawei},
  booktitle={KDD'23},
  pages={3458--3469},
  year={2023},
  organization={ACM}
}
```
