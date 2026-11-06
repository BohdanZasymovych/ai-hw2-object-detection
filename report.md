---
title: Homework 1 -- Your Title Here
author: John Doe
course: Artificial Intelligence · UCU
date: 2026-10-12
---

## Introduction

Write your report here using regular **Markdown**. This text is an example — replace it.

## Method

Inline math works with single dollars: $f(x) = \sigma(Wx + b)$. Display math uses double dollars:

$$
\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^{N} y_i \log \hat{y}_i
$$

Code blocks are highlighted:

```python
def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()
```

## Results

| Model    | Accuracy | F1   |
|----------|----------|------|
| Baseline | 0.71     | 0.68 |
| Ours     | 0.84     | 0.82 |

Images go into the `images/` folder:

<!-- ![Training loss](images/loss.png) -->

## Conclusion

> Summarize your findings here.

## Run instructions

**Code layout.** Shared code is in `utils/common.py`; each step has its own notebook in `notebooks/` (`01_data_analysis` … `06_final_pipeline`). Trained runs (metrics, predictions, checkpoints) are stored in the Hugging Face model repo `BohdanZasymovych/ai-hw2-object-detection` under `runs/<name>/`. All randomness is seeded with `SEED = 42`.

**Local setup.**

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The competition data is downloaded on the first run of `01_data_analysis` (Kaggle API token in `~/.kaggle/kaggle.json`). Tested with Python 3.14.4, torch 2.15.0, torchvision 0.29.2; minimum versions are in `requirements.txt` and are checked on import.

**Kaggle.** Create a notebook, attach the competition data, enable a GPU and Internet, and add the secret `HF_TOKEN` (Hugging Face write token). The first cell of each notebook clones this repository and installs only the missing packages; Kaggle's preinstalled torch/torchvision are used. Set `RUN` to the stages to train in that session; finished stages are uploaded to the Hugging Face repo.

**Reproducing the results without training.** Run a notebook locally with `RUN = set()`: all stages are downloaded from the Hugging Face repo and the tables, plots and failure cases are rebuilt from the saved metrics and predictions.
