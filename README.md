# Py-Torch Sequence Classification 
An LSTM-based architecture to train a model for Sequence Classification.

The purpose of the repo is to show that even a tiny LSTM with a tiny dataset can generalize well enough for small use-cases as it can be for DIY projects.

You may think of it as a glorified RegEx.
Not everything has to be a Large Language Model.

## Install
Through [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):
```bash
$ conda env create --file environment.yml
```

## How To Use (Training)
Train a new model using training.py:

By default creates a new model in ./models directory called "newestModel".

Each model is made of three files:
- model.pth: weights of the model
- model.json: the model configuration (embedding size, hidden size, ...)
- model.pkl: a pickle of the vocabularies used (these are built during training from the dataset)

Use '--help' for a complete list of parameters:

```bash
$ python training.py -h
usage: training.py [-h] [--path PATH] [--data DATA] [--lr LR] [--nepochs NEPOCHS] [--hidden_dim HIDDEN_DIM] [--embedding_dim EMBEDDING_DIM] [--dropout DROPOUT]

Train a new Sequence model.

options:
  -h, --help            show this help message and exit
  --path PATH           Model save path
  --data DATA           Path to dataset (plaintext)
  --lr LR               Learning rate
  --nepochs NEPOCHS     Number of epochs
  --hidden_dim HIDDEN_DIM
                        Size of hidden vector
  --embedding_dim EMBEDDING_DIM
                        Size of each embedding vector
  --dropout DROPOUT     Dropout value for regularization

```

## How To Use (Inference)
The inference script simply loads the model and applies it to a given string.

```bash
$ python inference.py --sentence "turn on the radio"
{'OBJ': 'radio', 'VERB': 'turn on'}
```

The output can be either 'clean' (as above) or raw.
Clean returns a dictionary where each key is a label and each value is the concatenation of the tokens classified as that label.

Raw just returns the list of labels for each token:

```bash
$ python inference.py --sentence "turn on the radio" --raw
['VERB', 'VERB', 'O', 'OBJ']
```
