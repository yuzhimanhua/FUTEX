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
Three datasets are used in our experiments. The paper titles, labels, and references come from the [MICoL](https://github.com/yuzhimanhua/MICoL) and [MAPLE](https://github.com/yuzhimanhua/MAPLE) projects. The paper abstracts and full texts come from the [S2ORC](https://github.com/allenai/s2orc) project. Dataset statistics are listed below.
|  | MAG-CS | PubMed | Art |
|--|--------|--------|-----|
| \# Papers (all for testing) | 96,718   | 251,573   | 328     |
| \# Labels                   | 10,909   | 16,070    | 1.990   |
| \# Words / Paper            | 4071.69  | 4901.42   | 5152.46 |
| \# Paragraphs / Paper       | 45.91    | 33.38     | 32.94   |
| \# Sections / Paper \*      | 13.80    | 10.90     | 7.01    |
| \# Labels / Paper           | 5.84     | 8.69      | 3.01    |

\*: Sections and subsections are not distinguished in S2ORC.

### Data Format

In each dataset folder (e.g., ```MAGCS/```), you can see three files: ```{dataset}_paper.json```, ```{dataset}_label.json```, and ```{dataset}_candidates.json```.

```{dataset}_paper.json``` contains the paper id, title, abstract, full text (sections and paragraphs), labels, and references.
```
{
  "paper": "2140839178",
  "s2orc_id": "2874113",
  "title": "high resolution ofdm channel estimation with low speed adc using compressive sensing",
  "abstract": "abstract-orthogonal frequency division multiplexing (ofdm) is a technique that will prevail in the next generation wireless communication ...",
  "paragraphs": [
    {
      "section": "abstract (0)",
      "text": "abstract-orthogonal frequency division multiplexing (ofdm) is a technique that will prevail in the next generation wireless communication ..."
    },
    {
      "section": "i. introduction (0)",
      "text": "orthogonal frequency division multiplexing (ofdm) has been widely applied in wireless communication systems ..."
    },
    {
      "section": "i. introduction (0)",
      "text": "some channel estimation schemes proposed in literature are based on pilots, which form the reference signal used by both the transmitter and the receiver ..."
    },
    ...
  ],
  "label": [
    "73836528", "185429906", "156996364", "76155785", ...
  ],
  "reference": [
    "2151730221", "2133698785"
  ]
}
```

```{dataset}_label.json``` contains the label id, name(s), and definition.
```
{
  "label": "10389098",
  "name": [
    "batch file"
  ],
  "definition": "a batch file is a kind of script file in dos, os 2 and microsoft windows ...",
  "combined_text": "batch file. a batch file is a kind of script file in dos, os/2 and microsoft windows ..."
}
```
**NOTE: Each label can have more than one name (e.g., PubMed) or an empty definition (e.g., Art). Please refer to the file in the corresponding dataset for its format.**

```{dataset}_candidates.json``` contains the labels whose name(s) appear in a paper's title/abstract. Such labels are considered as initial candidates for classification.
```
{
  "paper": "2140839178",
  "matched_label": [
    "26668531", "176012381", "124851039", "16885038", "47798520", ...
  ]
}
```

## Running on New Datasets
To run our model on new datasets, you need to prepare the following things:

(1) Create a new dataset folder ```{dataset}/```.

(2) The paper file ```{dataset}/{dataset}_paper.json```. **NOTE: If you do not have paper full texts, leave the ```paragraphs``` field an empty list AND set ```--full_text``` as 0 in [```./run.sh```](https://github.com/yuzhimanhua/FUTEX/blob/main/run.sh#L15). Titles and abstracts are required.**

(3) The label file ```{dataset}/{dataset}_label.json```. **NOTE: If you do not have label definitions, leave the ```definition``` field an empty string AND put only label name(s) into the ```combined_text``` field. Label name(s) are required.**

(4) The candidate file ```{dataset}/{dataset}_candidates.json```. It should be easily obtained by exact name matching.

## Citation
If you find this repository useful, please cite the following paper:
```
@inproceedings{zhang2023weakly,
  title={Weakly Supervised Multi-Label Classification of Full-Text Scientific Papers},
  author={Zhang, Yu and Jin, Bowen and Chen, Xiusi and Shen, Yanzhen and Zhang, Yunyi and Meng, Yu and Han, Jiawei},
  booktitle={KDD'23},
  pages={3458--3469},
  year={2023}
}
```
