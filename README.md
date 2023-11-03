# Practical Machine Learning and Deep Learning - Assignment 1 - Text De-toxification

## Personal information

Author: Mark Zakharov
<br>
Email: ma.zakharov@innopolis.university
<br>
Group: BS20-DS

## Task description

Text Detoxification Task is a process of transforming the text with toxic style into the text with the same meaning but with neutral style.

> Formal definition of text detoxification task can be found in [Text Detoxification using Large Pre-trained Neural Models by Dale et al., page 14](https://arxiv.org/abs/2109.08914)

I create a solution for detoxing text with high level of toxicity.

## Data Labeling

Level of Toxicity is labeled with annotating binary classification by people. Text is passed to annotators for them to put specific label toxic/non-toxic. Then number of positive / toxic assesments are divided by the total number of annotators. This process is performed for every entry in the data, resulting in toxicity dataset.

By this process, we have text with toxicity level. However, for training the model it is best to have sample with high toxicity level and its paraphrazed version with low toxicity level. This gives an opportunity for the model to distiguish from the overall meaning of the text and concentrate on decreasing the level of toxicity (dirung the training process). That is why the data that is provided has the pair structure. Dataset structure is described in next section.

## Data Description

The dataset is a subset of the ParaNMT corpus (50M sentence pairs). The filtered ParaNMT-detox corpus (500K sentence pairs) can be downloaded from [here](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip). This is the main dataset for the assignment detoxification task.

The data is given in the `.tsv` format, means columns are separated by `\t` symbol.

| Column | Type | Discription | 
| ----- | ------- | ---------- |
| reference | str | First item from the pair | 
| ref_tox | float | toxicity level of reference text | 
| translation | str | Second item from the pair - paraphrazed version of the reference|
| trn_tox | float | toxicity level of translation text |
| similarity | float | cosine similarity of the texts |
| lenght_diff | float | relative length difference between texts |

## Structure of the repository

```
text-detoxification
├── README.md # The top-level README
│
├── data 
│   ├── external # Data from third party sources
│   ├── interim  # Intermediate data that has been transformed.
│   └── raw      # The original, immutable data
│
├── models       # Trained and serialized models, final checkpoints
│
├── notebooks    #  Jupyter notebooks. Naming convention is a number (for ordering),
│                   and a short delimited description, e.g.
│                   "1.0-initial-data-exporation.ipynb"            
│ 
├── references   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures  # Generated graphics and figures to be used in reporting
│
├── requirements.txt # The requirements file for reproducing the analysis environment, e.g.
│                      generated with pip freeze › requirements. txt'
└── src                 # Source code for use in this assignment
    │                 
    ├── data            # Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── models          # Scripts to train models and then use trained models to make predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │   
    └── visualization   # Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```


In the `README.md` file, you’ll find my name, email, and group number. Additionally, it contains basic information on how to use this repository.

The `reports` directory contains two reports about my work. In the **first report** describes my journey in creating the solution. It includes the architectures, ideas, problems, and data that led to my final solution. In the **second report** provides a detailed description of my final solution.

The `notebooks` directory contains two notebooks. **First notebook** includes my initial data exploration and the basic ideas behind data preprocessing. **Second notebook** provides information about the training and visualization of the final solution

The `src` directory contains all the code used for the final solution. This includes a script for creating intermediate data in `src/data/`, `train` and `prediction` scripts in `src/models`, and a visualization script in `src/visualization/`.

