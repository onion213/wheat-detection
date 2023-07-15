# Wheat Detection

Challenge kaggle.

ref: https://www.kaggle.com/competitions/global-wheat-detection

## Installation
```bash
$ poetry install
```
## Dataset
- download

```bash
$ poetry run kaggle competitions download -p data/ global-wheat-detection
$ unzip data/global-wheat-detection.zip -d data
```

- details
ref: https://arxiv.org/abs/2005.02162

## Training
```
$ poetry run python wheat_detection/train.py
`
