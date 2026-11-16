---
title: Homework 2 -- Object Detection
author: Bohdan Zasymovych
course: Artificial Intelligence · UCU
date: 2026-11-16
---


## Summary

The task is detecting 8 object classes in video frames, scored by COCO mAP. Most objects are small, and scenes are dense. I trained four detector families under the same protocol: a sequence-level train/val split and the same budget for each. The query-based RT-DETR beat the CNN detectors (YOLO11, Faster R-CNN, RetinaNet) at every object size, so it became the base of the final pipeline. Three changes gave the largest gains: a longer, higher-resolution RT-DETR run, tile-based training for small objects, and a WBF ensemble of models whose errors differ. The final submission is a WBF ensemble of two RT-DETR models at different input sizes and YOLO11s, retrained on train + val.

## 1. Data understanding and preprocessing

Source: ![01_data_analysis.ipynb](notebooks/01_data_analysis.ipynb)

The dataset contains videos and labels for them with [N] classes and ignore areas. It is split into train ([N] videos), validation ([N] videos) and test. This split is kept the same for all models they are trained on the trains and val is used only for their validation and metrics, test is not labeled and is used for scoring on kaggle (only final models were trained on both train and val and then submitted). The dataset also has ignored regions during training boxes in them are jsut removed and during validation any detection in this regions doesnt count as mistake. Validation dataset is not very representable since it contains only 7 videos and additionaly class distribution is unbalanced

Sample frames with labels:

[add 2 frames, the ones for which i noted incorrect labels]

The dataset (both train and val splits) is unbalanced, it contains much more images of the person, car and motor then of all other classes, especially bus, it is absent from 32 out of 56 train sequences. [add share of the classes in the parenthesis to the text above]

[put plot with class share]


Image resolution and object sizes:

dataset contains images of different resolutions from 680x382 to 3840x2160. Train has most images with medium resolution (1344x756 - 1920x1080), val has with higher (1904x1071 - 3840x2160) and test with lower (680x382 - 1360x764)
and val has approximatelly the same distribution of resolutions while test has more low resolution images and no images of resolution higher than 1920x1080.

Most object are small [share %] and medium [share %] size (COCO bucket at 1280, [put exact pizel sizes]). By classes person, bicicle and motor are mostly small, car, van, truck, tricicle are mostly medium and bus is mostly medium and big. The median size of the object per sequence varies from 18 to 58px

For the training and inference models must get images of the fixed size. To descide on this size i checked the share of the objects which long sise will be <16px after resizing (very hard to clasify). Descided to use resolution of 1280 (long side) since for it share is 9.8% and for 960 its 24% so 1280 is good midleground between training/inference time and model performance

[put tables with percent of the boxes of the sizes and with percent of each size in each class]

Since some models which were used are anchor based models which uses anchors with a few predefined box sizes for the predictions and then refines size I descided to to try to find a few best sizes. To do so i runned k-means (with k = 3, 5, 9) on the box sizes, I got next sizes which are used later for the training.

[put table with anchor sizes]

Scene density:

it was found that most scenes are quite dense and contain a lot of object. Mean number of object varies from 5 to 143 accross sequences

[put table from 2.6 scene density and leave only mean, median and percent of the frames with > 100 objects]


Temporal redundancy:

To check whether there is temporal redundancy in the frames (framerate is high and a few frames in the sequence are almost the same) because there are almost dublicating frames they almost do not contribute to the model performance but they slow training time. I tried different gap (k) of frames and find the median IoU of the same boxe k frames appart. From the finding i descided to leave only every 3-rd frame since it povides balance between number of the frames left in the dataset which impacts train time and removing too many frames and losing information.

[put table from section 2.7]

Cross validation:
Cross-validation is not used. K-fold CV would multiply the training time by approximatelly K, and neither my local GPU (RTX 2050; 4GB VRAM) nor Kaggle's 30 h weekly GPU limit is enough to train every model K times. All models are instead trained once on the same sequence-level train/val split with the same seed, so results are comparable, but differences of a few thousandths of mAP are within noise (the val set has only 7 videos).

Summary:

[add short list of the conclussions, main descisions]

## 2. Validation protocol

> **Contains:** short and precise, this is the base every comparison relies on.
> - Split by sequence (GroupKFold / fixed sequence-level split), no frame leakage; explain if a single fold is used instead of full CV because of compute budget and why that is acceptable.
> - Metric: COCO mAP@[.5:.95], AP50, AP_small/medium/large via pycocotools/faster-coco-eval, ignored regions as crowd.
> - Fixed training budget for the required comparisons (epochs / image size / batch / GPU) and how inference speed is measured (hardware, batch size, FPS or ms/img).
> - How the public LB relates to val (does val ranking agree with LB?).

## 3. Detector families

> **Contains:** one subsection per model, same template each time so they are easy to compare:
> setup (1–2 lines) → experiments table (baseline, exp 1, exp 2, tuned) → required theoretical question (2–3 sentences) → failure cases (3 bullets with cause) → one-line takeaway.
> Put the long explanations into the notebooks and link to them; the report keeps only the conclusion.

### 3.1 Two-stage: Faster R-CNN

> - Setup: torchvision Faster R-CNN, backbone, pretrained weights, schedule.
> - Experiments: RPN proposals × NMS IoU grid (`frcnn_grid`), anchor sizes (`frcnn_anchors`), tuned (`frcnn_final`). Table with mAP / AP50 / AP_s / FPS.
> - What RPN vs. RoI head contribute (proposal recall vs. classification/refinement — use the numbers from section 8 of the notebook).
> - 3 failure cases with plausible causes (tiny objects missed by RPN, dense crowds suppressed by NMS, confusion of similar classes...).

### 3.2 One-stage anchor-based: RetinaNet

> - Experiments: anchor sizes grounded in the size statistics from section 1 (`retina_anchors`), IoU thresholds of anchor assignment (`retina_iou`), tuned (`retina_final`).
> - Does removing the proposal stage cause issues: foreground/background imbalance, handled by focal loss; what you saw in practice.
> - Comparison initial vs. tuned vs. tuned Faster R-CNN.

### 3.3 One-stage anchor-free: YOLO11

> - How locations are assigned without anchors (task-aligned assigner), effect on small/dense objects.
> - Experiments: P2 high-resolution head (`yolo_p2`), NWD box loss (`yolo_loss`), tuned test-time resolution × NMS IoU (`yolo_final`). State the search range and the selected values.
> - Comparison with tuned anchor-based models broken down by object size (AP_s/AP_m/AP_l). First public LB score.

### 3.4 Query-based: RT-DETR

> - Experiments: separate backbone/transformer LR (`rtdetr_lr`), denoising queries count (`rtdetr_nd200/400`), tuned (`rtdetr_final`).
> - Bipartite matching removes NMS — 2 sentences, and how it changes training (slower convergence, one-to-one targets).
> - Plot: val mAP over epochs vs. a CNN detector under the same budget. State whether it converged.
> - Small objects and dense scenes vs. YOLO.

### 3.5 Comparison of all models

> - One summary table: every tuned model (+ baselines in gray or separate rows): mAP, AP50, AP_s, AP_m, AP_l, FPS, public LB.
> - 3–4 sentences: which family wins on this data and why (small objects, density), the speed/accuracy trade-off, and which models go into the final pipeline.

## 4. Final pipeline

> **Contains:** must clearly separate "data/inference changes" from "model changes" (graded separately). Keep the same validation as above.

### 4.1 Data and inference strategy

> Source: `notebooks/final_pipeline/01_data_tests.ipynb`.
> - Table: reference vs. masked ignore regions (`s1_ignore`), low-resolution simulation (`s1_lowres`), less mosaic (`s1_mosaic`), tile-based training (`s1_tile`) — val mAP and public LB.
> - Which changes helped, which did not, and a likely reason for each.

### 4.2 Model changes

> Source: `notebooks/final_pipeline/02_long_runs.ipynb`.
> - Long runs at 960 and 1280 (`s2_960`, `s2_1280`): gain from more epochs / higher resolution compared to the budget-limited runs in section 3.
> - Effect of model change kept separate from the data change in 4.1.

### 4.3 Ensembles and final selection

> Source: `notebooks/final_pipeline/03_final_models.ipynb`.
> - WBF ensemble options (1280 + 960 + YOLO11s + tile) — table with val mAP and public LB.
> - Retraining on train + val: why it is allowed now (hyperparameters already fixed) and what changed.
> - The two submissions selected for the private LB and why (internal val, not only public LB).

### 4.4 Unsuccessful attempts

> Short list of what did not work, with the numbers: e.g. NWD loss, low-resolution simulation, specific ensemble members, any bugs found (WBF fix). One line each: what, result, likely why.

## 5. Conclusions

> 3–5 bullets: main lessons (resolution and small objects matter most, which family fits VisDrone best, what data/inference strategy gave the largest gain, what you would try next with more compute — e.g. temporal post-processing, SAHI).

## 6. Final model and run instructions

> **Contains:** name the exact final model/ensemble and the submission file. Then how to reproduce. Draft from the template version below — update notebook names and add the final-pipeline steps.

**Final model.** _TODO: ensemble name, members, input sizes, WBF parameters, submission file in `submissions/`._

**Code layout.** Shared code is in `utils/common.py` and `utils/ultra.py`; each step has its own notebook: `notebooks/01_data_analysis` … `05_query_based` for the required detectors and `notebooks/final_pipeline/01_data_tests`, `02_long_runs`, `03_final_models` for the final pipeline. Trained runs (metrics, predictions, checkpoints) are stored in the Hugging Face model repo `BohdanZasymovych/ai-hw2-object-detection` under `runs/<name>/`. All randomness is seeded with `SEED = 42`.

**Local setup.**

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The competition data is downloaded on the first run of `01_data_analysis` (Kaggle API token in `~/.kaggle/kaggle.json`). Tested with Python 3.14.4, torch 2.15.0, torchvision 0.29.2; minimum versions are in `requirements.txt` and are checked on import.

**Kaggle.** Create a notebook, attach the competition data, enable a GPU and Internet, and add the secret `HF_TOKEN` (Hugging Face write token). The first cell of each notebook clones this repository and installs only the missing packages; Kaggle's preinstalled torch/torchvision are used. Set `RUN` to the stages to train in that session; finished stages are uploaded to the Hugging Face repo.

**Reproducing the results without training.** Run a notebook locally with `RUN = set()`: all stages are downloaded from the Hugging Face repo and the tables, plots and failure cases are rebuilt from the saved metrics and predictions.

> _TODO: add the exact sequence to regenerate the submitted CSV (which notebook, which `RUN` stages, expected runtime on Kaggle T4)._

## References

> Every paper/library you took ideas or code from (assignment requires citing): VisDrone, Faster R-CNN, FPN, RetinaNet, YOLO11/Ultralytics, TOOD, NWD, RT-DETR, DN-DETR, WBF, pycocotools/faster-coco-eval, SAHI if used.
