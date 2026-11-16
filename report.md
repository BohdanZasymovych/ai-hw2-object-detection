---
title: Homework 2 -- Object Detection
author: Bohdan Zasymovych
course: Artificial Intelligence · UCU
date: 2026-11-16
---


## Summary

The task is detecting 8 object classes in video frames, scored by COCO mAP. Most objects are small, and scenes are dense. I trained four detector families under the same protocol: a sequence-level train/val split and the same budget for each. The query-based RT-DETR beat the CNN detectors (YOLO11, Faster R-CNN, RetinaNet) at every object size, so it became the base of the final pipeline. Three changes gave the largest gains: a longer, higher-resolution RT-DETR run, tile-based training for small objects, and a WBF ensemble of models whose errors differ. The final submission is a WBF ensemble of two RT-DETR models at different input sizes, a tile-based RT-DETR and YOLO11s.

## 1. Data understanding and preprocessing

Source: [01_data_analysis.ipynb](notebooks/01_data_analysis.ipynb)

### 1.1 Dataset and split

The dataset contains drone videos with box labels for 8 classes (person, bicycle, car, van, truck, tricycle, bus, motor) and ignore regions. It is split by video, so no video appears in two splits.

| split | videos | frames | boxes |
|---|---|---|---|
| train | 56 | 24,201 | 1,105,516 |
| val | 7 | 2,846 | 114,132 |
| test | 48 | 3,811 | hidden |

All models are trained on train, and val is used only for validation and metrics. Test is unlabeled and scored on Kaggle; only the final models were trained on train + val. The validation set is not very representative: it has only 7 videos and a different class mix (1.4), so per-class metrics on val are noisy.

### 1.2 Ignore regions

Ignore regions are areas where objects are not labeled. In training they are not used as targets; in validation, a detection inside them does not count as a mistake. 75% of train frames have at least one ignore region, but only 1.1% of boxes lie inside one, so objects there are mostly not annotated at all.

### 1.3 Label quality

Some frames have unlabeled objects. In the val frame below, not all cars in the parking lot (top right) are labeled. There are also incorrect labels, for example tricycle and motor boxes on the tree canopy.

![Val frame #127 of uav0000182_00000_v: unlabeled parked cars and boxes on the tree canopy](images/missing_labels.jpg)

One train frame has many incorrect labels: in the bottom right corner of #322 there are many boxes with no object inside, while its neighbours in the video do not have them (180 boxes against 100 in #321 and 88 in #323).

![Two consecutive train frames: #321 and #322 of uav0000295_02300_v](images/sample_frames.jpg)

### 1.4 Class distribution

The dataset is unbalanced: car (46% of train), person (30%) and motor (9%) dominate, the other classes have 3–4% each, and bus only 0.9%; it is absent from 32 of the 56 train videos. Val has a different mix: more person (44%), less car (28%) and only 264 buses (0.2%).

![Boxes per class and class share per video](images/class_distribution.png)

### 1.5 Image resolution and object sizes

Resolutions range from 680x382 to 3840x2160. Most train images are medium-sized (53% are 1904x1071), val has more high-resolution ones (34% are 3840x2160), and test has none above 1920x1080 but 16% low-resolution ones (680x382, 960x540) that train and val lack.

Most objects are small (54%) or medium (43%), measured as $\sqrt{w \cdot h}$ at a 1280 px long side with the COCO buckets (small $< 32$ px, large $> 96$ px). Person, bicycle and motor are mostly small, car, van, truck and tricycle mostly medium, and bus medium and large. The median object size per video varies from 18 to 58 px.

| % of boxes | small | medium | large |
|---|---|---|---|
| **all classes** | **54** | **43** | **3** |
| person | 73 | 27 | 0 |
| bicycle | 70 | 30 | 0 |
| car | 42 | 53 | 4 |
| van | 44 | 51 | 5 |
| truck | 26 | 59 | 14 |
| tricycle | 36 | 60 | 4 |
| bus | 13 | 55 | 32 |
| motor | 70 | 29 | 0 |

To choose the input size, I checked the share of objects below 16 px after resizing, since these are very hard to detect. I chose a 1280 px long side (9.8%, against 24% at 960 px) as a middle ground between training/inference time and performance.

| long side, px | 640 | 960 | **1280** | 1920 |
|---|---|---|---|---|
| % of boxes < 16 px | 54.2 | 24.0 | **9.8** | 1.7 |

### 1.6 Anchor sizes and shapes

Anchor-based models (Faster R-CNN, RetinaNet) predict boxes relative to a few predefined anchor sizes and shapes. To find good sizes, I ran k-means ($k = 3, 5, 9$) on the box widths and heights at 1280 px:

| k | mean best IoU | anchors (w x h, px) |
|---|---|---|
| 3 | 0.60 | 15x21, 32x32, 75x58 |
| 5 | 0.66 | 13x18, 19x34, 32x23, 44x41, 92x66 |
| 9 | 0.73 | 11x15, 13x28, 22x19, 20x39, 40x22, 30x31, 39x53, 63x34, 100x72 |

These shapes only show the useful size range, about 13–100 px, far below the 32–512 px default. torchvision takes one anchor size per feature-pyramid level combined with fixed aspect ratios, so the training sizes come from a 1D k-means ($k = 5$, log scale) of $\sqrt{w \cdot h}$: 13/21/31/48/87 px for Faster R-CNN, and 13/26/52/104/208 px for RetinaNet (the smallest size, doubled per level like the stride). The aspect ratios stay at the default $\{0.5, 1, 2\}$, since they cover the box ratios $h/w$:

| percentile | 10th | 50th | 90th |
|---|---|---|---|
| $h/w$ | 0.47 | 1.06 | 2.25 |

### 1.7 Scene density

Most scenes are dense. The mean number of objects per frame varies from 5 to 143 across videos, and the densest frame has 180, so a model must output at least 300 detections (or DETR queries) per image.

| | train | val |
|---|---|---|
| mean objects per frame | 45.7 | 40.1 |
| median | 43 | 37 |
| frames with > 100 objects | 7.6% | 6.4% |

### 1.8 Temporal redundancy

The frame rate is high, so consecutive frames are near-duplicates: they add little to performance but slow down training. I measured the median IoU of the same object $k$ frames apart:

| frame gap $k$ | 1 | 2 | **3** | 5 | 10 |
|---|---|---|---|---|---|
| median IoU of the same object | 0.90 | 0.81 | **0.73** | 0.60 | 0.36 |
| % of objects with IoU > 0.5 | 96 | 86 | **75** | 59 | 39 |

I keep every 3rd frame (8,090 train / 951 val frames), a balance between training time and losing information.

### 1.9 Preprocessing

| step | setting |
|---|---|
| frames | every 3rd frame of each video |
| resize | long side 1280 px, aspect ratio kept |
| normalisation | pixels scaled to [0, 1]; mean/std are applied inside each model |
| training augmentation | horizontal flip ($p = 0.5$), scale jitter 0.75–1.25, colour jitter 0.2 (YOLO and RT-DETR: Ultralytics' built-in augmentation) |
| val / test | resize only |
| targets | 8 classes; ignore regions are not targets and stay in the evaluation ground truth as `iscrowd` |
| reproducibility | seed 42 for python, numpy, torch and the data loaders |

### 1.10 Summary

- **Small objects:** 54% of boxes are under 32 px → 1280 px input, anchors from 13 px, high-resolution feature levels.
- **Dense scenes:** up to 180 objects per frame → at least 300 detections / queries per image.
- **Class imbalance:** car and person are 76% of boxes, bus under 1% → rare-class AP is noisy.
- **Noisy labels:** unlabeled objects and some false boxes cap the achievable mAP.
- **Near-duplicate frames:** every 3rd frame is kept, 3x faster training.
- **Val differs from test:** 4K frames only in val, low-resolution frames only in test → addressed in the final pipeline (section 4).

## 2. Validation protocol

- **Split:** the provided video-level split (56 train / 7 val videos), every 3rd frame, the same for all models.
- **No cross-validation:** K-fold CV would multiply training time by about K, which neither my local GPU (RTX 2050, 4 GB) nor Kaggle's 30 h weekly GPU limit allows. All models use the same split and seed (42), so results are comparable, but differences of a few thousandths of mAP are within noise.
- **Metrics:** COCO mAP@[.5:.95], AP50, AP_small / medium / large and AR (pycocotools), up to 300 detections per image; ignore regions stay in the ground truth as `iscrowd`.
- **Budget:** each required detector is trained from COCO-pretrained weights for about 1–1.5 h on a Kaggle T4, and its experiments keep the same budget. The input is 1280 px, except RT-DETR, which trains at 960 px because a 1280 px input costs 1.6x more per image.

## 3. Detector families

### 3.1 Two-stage: Faster R-CNN

Source: [02_two_stage.ipynb](notebooks/02_two_stage.ipynb)

#### Setup

Faster R-CNN with a ResNet-50 FPN backbone, pretrained on COCO, with a new 8-class head. SGD (momentum 0.9, weight decay 1e-4), lr 0.005 with $\times 0.1$ after epoch 2, 3 epochs, batch 4, 1280 px; the best epoch is kept.

#### Baseline

With the torchvision defaults (anchors 32–512 px, 1000 proposals, NMS IoU 0.5):

| mAP | AP50 | AP_small | AP_large | ms/img (T4) |
|---|---|---|---|---|
| 0.176 | 0.369 | 0.079 | 0.348 | 76 |

#### Experiment 1: RPN proposals × NMS IoU

Both are test-time settings, so no retraining is needed:

- **RPN proposals:** the number of top-scoring candidate regions the Region Proposal Network passes to the second stage (300, 1000, 2000).
- **NMS IoU:** non-maximum suppression keeps the model from giving several boxes for one object. It is applied after the second stage: if two predictions of the same class overlap with IoU above the threshold, only the more confident one is kept (0.4, 0.5, 0.6).

| mAP | NMS 0.4 | NMS 0.5 | NMS 0.6 |
|---|---|---|---|
| 300 proposals | 0.172 | 0.172 | 0.172 |
| 1000 proposals | 0.175 | 0.176 | **0.176** |
| 2000 proposals | 0.175 | 0.174 | 0.171 |

The grid is almost flat: all mAP changes are within 0.005, which is likely noise. The best configuration is 1000 proposals and NMS 0.6.

#### Experiment 2: anchor sizes

The default anchors (32–512 px) are replaced by the k-means sizes 13/21/31/48/87 px, one per FPN level, with the same aspect ratios 0.5, 1, 2 (see [1.6](#16-anchor-sizes-and-shapes)). To check how well they fit, I matched each train box with every anchor shape and kept the best IoU:

| box size | % of boxes | best IoU, default | best IoU, k-means |
|---|---|---|---|
| $< 32$ px | 53.9 | 0.48 | 0.77 |
| 32–96 px | 43.1 | 0.70 | 0.75 |
| 96–150 px | 2.5 | 0.73 | 0.61 |
| $\ge 150$ px | 0.5 | 0.60 | 0.25 |

The fit improves for objects under 96 px (97% of boxes) but worsens for larger ones, which are bigger than the largest anchor. Despite the better fit, mAP decreased by 0.027. Two likely reasons:

- **Large objects lost their anchors:** AP_large fell from 0.348 to 0.277, while AP_small did not change.
- **Pretraining was lost:** the COCO-pretrained RPN head was trained for the default sizes, and 3 epochs were not enough to adapt it to the new anchors.

#### Final model

Default anchors, 1000 proposals and NMS 0.6:

| model | mAP | AP50 | AP_small | AP_medium | AP_large | ms/img (T4) |
|---|---|---|---|---|---|---|
| baseline (NMS 0.5) | 0.176 | 0.369 | 0.079 | 0.166 | 0.348 | 76 |
| **final (NMS 0.6)** | **0.176** | **0.366** | **0.079** | **0.166** | **0.347** | **76** |
| k-means anchors | 0.149 | 0.334 | 0.081 | 0.152 | 0.277 | 76 |

![Val mAP per epoch and per training minute](images/frcnn_training.png)

The model had not converged: val mAP was still rising at epoch 3, so a longer schedule is the most likely gain.

#### What each stage contributes

- **RPN (first stage):** class-agnostic detection of objects. It scores anchors as object or background, roughly corrects their boxes and keeps the best candidates. Its recall is an upper bound for the whole model, since an object without a proposal cannot be detected.
- **RoI head (second stage):** pools features for each proposal, classifies it into the 8 classes or background, refines the box a second time and removes duplicates with NMS. It turns 1000 proposals into about 59 confident boxes per image.

| recall ($\text{IoU} \ge 0.5$) | small | medium | large |
|---|---|---|---|
| RPN, top 1000 proposals | 0.71 | 0.82 | 0.99 |
| final detections ($\text{score} \ge 0.3$) | 0.53 | 0.67 | 0.95 |

Small objects are limited by both stages: the RPN misses 29% of them, and the second stage loses another 18 points. More proposals do not help, since 2000 proposals give the same recall as 1000.

#### Failure cases

At $\text{score} \ge 0.3$ and $\text{IoU} \ge 0.5$:

1. **Small and occluded objects are missed.** Only 46% of small objects are detected correctly (24% missed, 23% mislocalized), against 81% of large ones, and 61% of heavily occluded objects are missed. Small objects have few pixels and weak features, and occluded ones look like background.
2. **Similar classes are confused.** Of the detected bicycles, 45% are classified as motors, possibly because motors are far more frequent in the dataset. Of the detected buses, 71% are classified as cars, a direct consequence of the imbalance and the very low number of buses.
3. **Missing labels and background false positives.** A third of the detections are on background; part of them are real objects without a label.

![Confusion matrix of the final model](images/frcnn_confusion.png)

Example frames, ground truth on top and predictions below:

![#127 of uav0000182_00000_v: missed small objects and unlabeled parked cars](images/frcnn_fail_missed_small.jpg)

*The frame from [1.3](#13-label-quality): not all parked cars are labeled, and the model detects them correctly, but its score is lowered because the GT does not have them. Some small objects are also misclassified or missed.*

![#190 of uav0000137_00458_v: bicycles classified as motors](images/frcnn_fail_confused.jpg)

*Many bicycles are classified as motors.*

![#211 of uav0000182_00000_v: background false positives](images/frcnn_fail_background.jpg)

*The same issues as in the first frame; in addition, a traffic light is classified as a van, and some other objects are misclassified or missed.*

### 3.2 One-stage anchor-based: RetinaNet

Source: [03_one_stage_anchor.ipynb](notebooks/03_one_stage_anchor.ipynb)

#### Setup

RetinaNet ResNet-50 FPN v2, pretrained on COCO, new 8-class layer. Levels P3–P7 (strides 8–128), 9 anchors per location, focal loss ($\alpha = 0.25$, $\gamma = 2$), GIoU box loss. SGD, lr 0.0025 with $\times 0.1$ after epoch 2, 3 epochs, batch 4, 1280 px. The baseline uses the torchvision defaults: anchor base sizes 32–512 px, IoU thresholds 0.5/0.4, NMS 0.5.

#### Experiment 1: anchor sizes

The base sizes start from the smallest k-means size (13 px, [1.6](#16-anchor-sizes-and-shapes)) and double up to 208 px; each level multiplies its base by $(1, 2^{1/3}, 2^{2/3})$, so the levels cover 13–330 px without gaps. Doubling follows the stride, keeping the same anchor/stride ratio on every level, which the shared head needs.

| box size | % of boxes | best IoU, default | best IoU, data |
|---|---|---|---|
| $< 32$ px | 53.9 | 0.48 | 0.83 |
| $\ge 32$ px | 46.1 | 0.83 | 0.82 |

The fit improves strongly for small objects, but mAP decreased by 0.042:

- **Fewer positives:** at $1.6\times$ the stride (default $4\times$) anchors overlap less, so an image gets 189 positives instead of 628.
- **Pretraining lost:** the COCO head was trained for anchors $4\times$ the stride, and 3 epochs were not enough to adapt it.

#### Experiment 2: IoU thresholds

An anchor is positive if its IoU with a GT box is $\ge$ the foreground threshold and background if it is below the background threshold. Small boxes rarely reach IoU 0.5, so lowering the thresholds from 0.5/0.4 to 0.4/0.3 gives them more positives (1,695 instead of 628 per image). mAP still decreased by 0.024, and even AP_small fell (0.058 → 0.048): the extra positives are loose boxes, so the model learns to score poorly placed boxes highly.

NMS 0.6 at test time gave no gain over 0.5 (mAP 0.145 vs 0.146).

#### Final model and comparison with Faster R-CNN

No experiment beat the baseline, so it is the final model.

| model | mAP | AP50 | AP_small | AP_medium | AP_large | ms/img (T4) |
|---|---|---|---|---|---|---|
| Faster R-CNN, tuned | 0.176 | 0.366 | 0.079 | 0.166 | 0.347 | 76 |
| **RetinaNet, baseline = final** | **0.146** | **0.289** | **0.058** | **0.135** | **0.289** | **57** |
| data anchors | 0.104 | 0.246 | 0.046 | 0.095 | 0.227 | 56 |
| IoU 0.4/0.3 | 0.122 | 0.243 | 0.048 | 0.108 | 0.248 | 57 |

![Val mAP per epoch and per training minute](images/retina_training.png)

RetinaNet is 0.030 mAP below Faster R-CNN but 25% faster. It matches it on person (AP 0.220 vs 0.219) and loses most on rare large vehicles (bus 0.020 vs 0.088, truck 0.116 vs 0.193). It plateaued after the learning-rate drop, while Faster R-CNN was still improving.

#### What removing the proposal stage costs

- **Foreground/background imbalance:** about 170k anchors per frame against about 46 objects, so easy background dominates the loss. Addressed by focal loss (down-weights easy anchors) and the prior bias init; Faster R-CNN instead samples 512 RoIs with up to 25% positives.
- **Only one refinement:** class and box are predicted once from a fixed grid location. Addressed by IoU-based losses, quality-aware scores (centerness, GFL) or cascade refinement.
- **Small objects depend only on the anchor assignment**, with no stride-4 level and no second chance:

| covered ($\text{IoU} \ge 0.5$, $\text{score} \ge 0.3$) | small | medium | large |
|---|---|---|---|
| Faster R-CNN | 0.53 | 0.67 | 0.95 |
| RetinaNet | 0.33 | 0.56 | 0.89 |

- **Faster inference:** no per-proposal computation (57 vs 76 ms/img).

#### Failure cases

At $\text{score} \ge 0.3$ and $\text{IoU} \ge 0.5$:

1. **Small and occluded objects are missed:** 28% of small objects are correct (48% missed; Faster R-CNN 46% correct), and 70% of heavily occluded ones are missed.
2. **Similar classes are confused**, as in 3.1: 48% of detected bicycles become motors and 35% persons, all 49 detected buses become cars, and 58% of vans become cars.
3. **Background false positives:** 20% of detections. Duplicates are rare (1.5%, Faster R-CNN 4.6%), since each location is predicted once.

![Confusion matrix of the final model](images/retina_confusion.png)

Example frames, ground truth on top and predictions below:

![#193 of uav0000182_00000_v: missed small objects](images/retina_fail_missed_small.jpg)

*Many small objects are missed.*

![#88 of uav0000137_00458_v: bicycles missed](images/retina_fail_confused.jpg)

*Almost all bicycles are missed; only the persons riding them are detected.*

![#121 of uav0000339_00001_v: background false positives](images/retina_fail_background.jpg)

*Many boxes on the background, and some objects covered by several boxes.*

### 3.3 One-stage anchor-free: YOLO11

Source: [04_one_stage_anchor_free.ipynb](notebooks/04_one_stage_anchor_free.ipynb)

#### Setup

YOLO11s, pretrained on COCO, new 8-class head; DFL + CIoU box loss, BCE classification. AdamW, lr 0.001, nominal batch 64, Ultralytics augmentation (mosaic off for the last 2 epochs), 1280 px. The 1.1 h budget fits 9 epochs (Faster R-CNN: 3). Baseline: levels P3–P5, CIoU, NMS 0.7.

#### How locations are assigned without anchors

Each feature-map cell is a point that predicts the distances to the 4 box sides and 8 class scores. During training, the task-aligned assigner (TAL) takes the points inside a GT box as candidates and makes the top 10 by $t = \text{score}^{0.5} \cdot \text{CIoU}^{6}$ positives, so positives follow the current predictions instead of fixed anchor shapes. Measured on train frames:

| GT box size | candidate points | on the object |
|---|---|---|
| tiny ($< 16$ px), P3–P5 | 5.8 | 59% |
| small (16–32 px), P3–P5 | 12.4 | 97% |
| tiny ($< 16$ px), P2–P5 | 14.2 | 97% |

Tiny boxes have fewer candidates than the top 10, so all of them become positives, including those off the object, and in dense scenes neighbours compete for the same points (10% of tiny boxes share all of them).

#### Experiment 1: P2 level

Previous models struggled with small objects, so I added a P2 level (stride 4) to the default P3–P5 (strides 8, 16, 32). Its feature map (320×180 for a 1280×720 frame) preserves more detail and gives tiny boxes 2.4× more candidates (table above), but increases training time.

P2 improved AP_small (+0.014) and AR (+0.029), but mAP decreased by 0.007. Within the same time budget it reached only 5 epochs (baseline 9), and at epoch 5 both were level, so it may outperform the baseline with more epochs.

#### Experiment 2: NWD loss

IoU-based losses are very sensitive to small shifts of small objects. Normalized Wasserstein Distance models each box as a 2D Gaussian: $\text{NWD} = \exp(-W_2 / C)$ depends on the absolute pixel offset, not the box size, and stays informative without overlap. With $C = 32$ px (the median box size), the box loss becomes $0.5\,(1 - \text{CIoU}) + 0.5\,(1 - \text{NWD})$.

It decreased mAP by 0.011, and its val mAP peaked at epoch 5 and then fell. A likely cause: with $C = 32$, a 2 px shift gives NWD 0.92 at any box size, so half of the loss barely pushes for exact localization.

#### Final model and comparison

The best model (baseline) was tuned at test time over the input size (1280, 1536, 1920 px) and NMS IoU (0.6, 0.7). NMS changed mAP by at most 0.001. Higher resolution trades large objects for small ones: 1536 px raised AP_small (0.102 → 0.110) at equal mAP, and 1920 px lost large objects (AP_large 0.334 → 0.287), since the model was trained at 1280 px. **Selected: 1280 px, NMS 0.6.**

| model | mAP | AP50 | AP_small | AP_medium | AP_large | ms/img (T4) | public | private |
|---|---|---|---|---|---|---|---|---|
| Faster R-CNN, tuned | 0.176 | 0.366 | 0.079 | 0.166 | **0.347** | 76 | – | – |
| RetinaNet, tuned | 0.146 | 0.289 | 0.058 | 0.135 | 0.289 | 57 | – | – |
| YOLO11s baseline | 0.206 | 0.393 | 0.102 | 0.193 | 0.334 | 40 | – | – |
| P2 level | 0.199 | 0.386 | **0.116** | 0.179 | 0.366 | 41 | – | – |
| NWD loss | 0.195 | 0.384 | 0.104 | **0.194** | 0.308 | 41 | – | – |
| **YOLO11s final** | **0.207** | **0.398** | 0.102 | **0.194** | 0.334 | 41 | 0.2145 | 0.1757 |

![Val mAP per epoch and per training minute](images/yolo_training.png)

The curves show Ultralytics' own val mAP, which is not comparable with 3.1–3.2.

YOLO11s is the best and fastest model so far, ahead on small and medium objects; Faster R-CNN stays slightly ahead on large ones. Faster R-CNN still finds more objects (AR 0.401 vs 0.307), so YOLO wins by better localization (AP75 0.186 vs 0.151), not recall. Part of the gap may come from 9 vs 3 epochs.

#### Failure cases

At $\text{score} \ge 0.3$ and $\text{IoU} \ge 0.5$:

1. **Small objects are missed:** 35% correct (45% missed), between Faster R-CNN (46%) and RetinaNet (28%).
2. **Similar classes are confused:** van → car (54% of detected vans), bicycle → motor (41%), bus → car (55 of 69).
3. **Background false positives:** 16% of detections; duplicates are almost absent (0.4%).

![Confusion matrix of the final model](images/yolo_confusion.png)

Example frames, ground truth on top and predictions below:

![Missed small objects](images/yolo_fail_missed_small.jpg)

*Many small and distant objects are missed; some GT boxes lie on foliage with no visible object (as in [1.3](#13-label-quality)).*

![Class confusions](images/yolo_fail_confused.jpg)

*Distant cars are missed, and bicycles are classified as motors.*

![Background false positives](images/yolo_fail_background.jpg)

*The model performs well, but the GT is poor: most riders on motors are not labeled as persons, while the model detects them, and some GT boxes are on the background.*

### 3.4 Query-based: RT-DETR

Source: [05_query_based.ipynb](notebooks/05_query_based.ipynb)

#### Setup

RT-DETR-L (HGNetv2 backbone, hybrid encoder on P3–P5, 6 decoder layers, 300 queries, 32M parameters), pretrained on COCO. AdamW, lr 1e-4, weight decay 1e-4, nominal batch 16, the same augmentation and 1.1 h budget as YOLO. Ultralytics stretches the input to a square, so the model is trained at 960×960 (the same number of pixels as 1280×720); 1280 px costs 1.6× more per image and would fit fewer epochs. Only 4 epochs fit the budget. Baseline: one learning rate for the whole model, 100 denoising queries.

#### Bipartite matching and NMS

During training the Hungarian algorithm matches every GT box with exactly one query, the one with the lowest cost ($2\,\text{class} + 5\,L_1 + 2\,\text{GIoU}$); all other queries must predict "no object". A second query on the same object is unmatched, so the loss pushes its score down, and through self-attention the queries learn which of them keeps the object. Duplicates are removed by training, so NMS is not needed.

This makes training slower: each object gives one positive (YOLO: up to 10), and the target of a query can change between iterations while the predictions are still poor. Denoising queries help with this: noised copies of the GT boxes are passed to the decoder and must be reconstructed, with a known target and no matching.

#### Experiment 1: separate learning rates

Backbone lr 1e-5 and 1e-4 for the rest, as in the RT-DETR recipe. The pretrained backbone features already work, so a lower rate should protect them while the new layers and the matching settle. mAP increased by 0.014 (0.257 → 0.271), AP_small by 0.017 and AR by 0.055, in the same 4 epochs.

#### Experiment 2: denoising queries

The number of denoising groups is the number of denoising queries divided by the most boxes in an image of the batch. This dataset is much denser than COCO, so with the default 100 queries 83% of batches get only one group and denoising is almost off. I raised it to 200 and 400 (2.5 and 5.4 groups per batch). mAP stayed the same within noise (0.271, 0.269, 0.268); 4 epochs are probably too few to see the faster convergence denoising gives.

#### Final model and comparison

The best training (backbone lr 1e-5, 100 denoising queries) was tested at inputs of 800–1440 px. Unlike YOLO, it gains from a larger input than in training: 0.271 at 960 px, **0.282 at 1280 px**, 0.272 at 1440 px. Adding NMS after the model does not change mAP (0.282 with NMS 0.7) and only lowers AR (0.492 → 0.487).

| model | mAP | AP50 | AP_small | AP_medium | AP_large | AR | ms/img (T4) | public | private |
|---|---|---|---|---|---|---|---|---|---|
| Faster R-CNN, tuned | 0.176 | 0.366 | 0.079 | 0.166 | 0.347 | 0.401 | 76 | – | – |
| RetinaNet, tuned | 0.146 | 0.289 | 0.058 | 0.135 | 0.289 | 0.338 | 57 | – | – |
| YOLO11s, tuned | 0.207 | 0.398 | 0.102 | 0.194 | 0.334 | 0.307 | **41** | 0.2145 | 0.1757 |
| RT-DETR baseline, 960 px | 0.257 | 0.501 | 0.127 | 0.223 | 0.433 | 0.407 | 82 | – | – |
| backbone lr 1e-5, 960 px | 0.271 | 0.517 | **0.144** | 0.230 | 0.460 | 0.462 | 84 | – | – |
| denoising 200, 960 px | 0.269 | 0.515 | 0.142 | 0.233 | 0.453 | 0.440 | 83 | – | – |
| denoising 400, 960 px | 0.268 | 0.511 | 0.131 | 0.228 | 0.464 | 0.455 | 83 | – | – |
| **RT-DETR final, 1280 px** | **0.282** | **0.533** | 0.137 | **0.241** | **0.479** | **0.492** | 105 | **0.2590** | **0.2558** |

![Val mAP of RT-DETR and YOLO11s under the same budget](images/rtdetr_training.png)

RT-DETR is the best model so far, better at every object size and with much higher recall. It is also 3.5× larger and 2.6× slower than YOLO11s, so part of the gap comes from the model size. It is ahead of YOLO from the first epoch, but **it has not converged**: in every run the best epoch is the last one and mAP was still rising, while YOLO levelled off near 0.21.

#### Small objects and dense scenes

RT-DETR finds 61% of small objects (YOLO11s 39%, Faster R-CNN 53%), so its AP_small gain comes from fewer misses. It is also ahead in every density group, including frames with more than 100 objects (mAP 0.334 vs 0.277), and 300 queries are enough for 180 objects per frame. Matching does not remove all duplicates (2.0% of detections, YOLO with NMS 0.4%), but they have low scores, which is why NMS does not help.

#### Failure cases

At $\text{score} \ge 0.3$ and $\text{IoU} \ge 0.5$:

1. **Small objects are mislocalized:** 56% correct, but only 18% missed and 20% mislocalized; a few pixels give little signal for the box regression.
2. **Occlusion:** 57% of heavily occluded objects are missed, still the largest error (YOLO 71%).
3. **Many false positives:** 30% of detections are on background and 12% in ignore regions (YOLO 16% and 5%). It outputs 66 boxes per image against 31, so part of them are probably unlabeled objects.

The confusion matrix shows that this model partly fixes the classification of buses and bicycles: 40% of detected buses and 71% of bicycles are classified correctly, against 9% and 36% for Faster R-CNN. The same confusions remain, only smaller: van → car (28%), bicycle → motor (16%) and bus → van (52%).

![Confusion matrix of the final model](images/rtdetr_confusion.png)

Example frames, ground truth on top and predictions below:

![Missed small objects](images/rtdetr_fail_missed_small.jpg)

*A night frame: some distant objects are missed.*

![Class confusions](images/rtdetr_fail_confused.jpg)

*Most bicycles are classified as motors.*

![Background false positives](images/rtdetr_fail_background.jpg)

*Most "false positives" are real objects: riders on motors are detected as persons while the GT has only the motor box, and parked cars that have no labels are detected.*

## 4. Final pipeline

### 4.1 Data and inference strategy

Source: [final_pipeline/01_data_tests.ipynb](notebooks/final_pipeline/01_data_tests.ipynb)

All tests start from the best RT-DETR training (backbone lr 1e-5, 960 px, 1.1 h) and change one thing at a time. Besides val mAP I report **test-like mAP**, val without the frames wider than 1920 px, which the test set does not have. I tried four changes:

- **Masked ignore regions.** Objects inside ignore regions are mostly not labeled, so the model learns them as background. I painted the ignore regions grey in all train frames.
- **Low-resolution simulation.** The test set has low-resolution frames that train and val lack ([1.5](#15-image-resolution-and-object-sizes)). I downscaled 20% of the train frames to 680 or 960 px and enlarged them back, so some detail is lost. To measure it, val frames were degraded the same way.
- **Less mosaic.** Mosaic augmentation on half of the images instead of all, so the model also sees normal frames.
- **Tile-based model.** A separate model: frames are resized to 1920 px and cut into 6 overlapping 960 px tiles; each tile is predicted separately and the predictions are merged with NMS. Objects stay twice as large as in a 960 px frame, which should help small objects.

| change | val mAP | AP_small | test-like mAP | public | private |
|---|---|---|---|---|---|
| baseline (RT-DETR final) | 0.282 | 0.137 | 0.300 | 0.2590 | 0.2558 |
| masked ignore regions | 0.275 | 0.136 | 0.296 | **0.2693** | **0.2645** |
| low-resolution simulation | 0.286 | 0.137 | 0.300 | 0.2661 | 0.2584 |
| less mosaic (0.5) | 0.290 | 0.131 | 0.305 | 0.2624 | 0.2523 |
| tile model | **0.310** | **0.154** | **0.325** | 0.2629 | 0.2509 |
| baseline with tile inference only | 0.280 | 0.155 | 0.294 | – | – |

- **Masked ignore regions** lowered val mAP, but gave the best public and private scores.
- **Low-resolution simulation** keeps clean val unchanged and adds 0.012–0.017 mAP on degraded val frames; it also improved both leaderboard scores.
- **Less mosaic** improved val in all 7 videos, but barely changed the public score and lowered the private one.
- **The tile model** is the largest gain on val (+0.028, all 7 videos better), but alone it barely improved the public score (+0.004) and lowered the private one. Tile inference without tile training does not help, so the model must be trained on tiles.

Val and the leaderboard disagree here, and the differences are small, so I kept all three data changes together for the long runs (4.2). The tile model is kept as a separate model for the ensembles (4.3), since its errors differ from the full-frame models.

### 4.2 Model changes

Source: [final_pipeline/02_long_runs.ipynb](notebooks/final_pipeline/02_long_runs.ipynb)

The RT-DETR-L training curves were still rising when the 1.1 h budget ended (3.4). So I gave it 6 h per run to see its real performance, with the three data changes from 4.1. I also tried two training input sizes, 960 and 1280 px; each model is tested at a larger input than it was trained on, as in 3.4.

| model | epochs | val mAP | AP_small | AR | test-like mAP | public | private |
|---|---|---|---|---|---|---|---|
| RT-DETR final (3.4), 1.1 h | 4 | 0.282 | 0.137 | 0.492 | 0.300 | 0.2590 | 0.2558 |
| 960 px, tested at 1280 px | 21 | 0.299 | **0.150** | 0.488 | **0.310** | – | – |
| **1280 px, tested at 1440 px** | 13 | **0.306** | 0.149 | **0.500** | **0.310** | **0.3290** | **0.2763** |

![Val mAP of the long runs](images/long_runs_training.png)

Both models are better than the baseline, and the higher input resolution is better (+0.024 mAP, +0.070 public, +0.021 private). The best results were at epoch 7 in both runs; after that performance decreased because of overfitting, so most of the 6 h was not needed. The data changes were applied together, so their separate effects are not measured.

### 4.3 Ensembles and final selection

Source: [final_pipeline/03_final_models.ipynb](notebooks/final_pipeline/03_final_models.ipynb)

The models have different errors (RT-DETR has much higher recall than YOLO, the tile model is better on small objects), so I combined them with Weighted Boxes Fusion (WBF). WBF averages overlapping boxes of the same class from several models, weighted by score, and ranks boxes found by several models higher. All ensembles contain the 1280 px long run, the strongest single model.

| ensemble (train only) | val mAP | AP_small | test-like mAP | public | private |
|---|---|---|---|---|---|
| 1280 alone | 0.306 | 0.149 | 0.310 | 0.3290 | 0.2763 |
| 1280 + 960 | 0.311 | 0.150 | 0.313 | – | – |
| 1280 + 960 + YOLO11s | 0.323 | 0.166 | 0.330 | **0.3333** | 0.2801 |
| 1280 + 960 + tile | 0.330 | 0.164 | 0.335 | 0.3146 | 0.2841 |
| 1280 + YOLO11s + tile | 0.332 | 0.170 | 0.344 | 0.3224 | 0.2835 |
| **1280 + 960 + YOLO11s + tile** | **0.337** | **0.174** | **0.347** | 0.3290 | **0.2871** |
| 1280 + 960 + YOLO11s, train + val | – | – | – | 0.3037 | 0.2845 |

Every added model improves val, and the tile model gives the largest gain (+0.019 alone). On the public leaderboard the ranking is different: the best ensemble there has no tile model.

Finally, I retrained the 1280 px, 960 px and YOLO11s models on train + val with the same settings and a fixed number of epochs (8, 8 and 9, close to the best epochs found before), since all hyperparameters were already chosen. This version cannot be scored on val. On the leaderboard it was slightly better on private (0.2845 vs 0.2801) but worse on public (0.3037 vs 0.3333), so the gain from the extra data is small.

**Selected models:**

- **1280 + 960 + YOLO11s + tile (train only):** the best ensemble on val and on test-like val (0.337 / 0.347), the main criterion.
- **1280 + 960 + YOLO11s (train only):** the best public score (0.3333) and the best on val among the ensembles without the tile model.


## 5. Summary of results

| section | model | val mAP | AP_small | public | private |
|---|---|---|---|---|---|
| 3.1 | Faster R-CNN, tuned | 0.176 | 0.079 | – | – |
| 3.2 | RetinaNet, tuned | 0.146 | 0.058 | – | – |
| 3.3 | YOLO11s, tuned | 0.207 | 0.102 | 0.2145 | 0.1757 |
| 3.4 | RT-DETR-L, tuned (1.1 h) | 0.282 | 0.137 | 0.2590 | 0.2558 |
| 4.1 | RT-DETR-L, tile model | 0.310 | 0.154 | 0.2629 | 0.2509 |
| 4.2 | RT-DETR-L, long run, 1280 px | 0.306 | 0.149 | 0.3290 | 0.2763 |
| 4.3 | **WBF: 1280 + 960 + YOLO11s + tile** | **0.337** | **0.174** | 0.3290 | **0.2871** |

- **Small objects are the main difficulty.** Every gain came from handling them better: higher resolution, finer feature levels, tiles.
- **The query-based RT-DETR was the best family** at every object size, with much higher recall, but it is larger and slower than YOLO11s.
- **Changing the default anchors and assignment of the pretrained CNNs did not help** within 3 epochs; the pretrained settings were hard to beat.
- **Longer training with the data changes gave the largest single-model gain** (+0.070 public), and the ensemble of models with different errors gave the best overall result.

## 6. Run instructions

Install the dependencies with `pip install -r requirements.txt`. All randomness is seeded with `SEED = 42`.

**Run flags.** By default the notebooks train and submit nothing: `RUN = set()`, `SUBMIT = False`. Every stage is then downloaded from the Hugging Face repo `BohdanZasymovych/ai-hw2-object-detection`, and all tables and plots are rebuilt from the saved metrics and predictions.

**Training the final model from scratch** (on Kaggle):

First set `HF_REPO` in `utils/common.py` to your own Hugging Face repo and add an `HF_TOKEN` with write access; comparison cells that need untrained stages will fail, which does not affect the models.

1. `04_one_stage_anchor_free`: `RUN = {"baseline", "final"}` (about 1.5 h).
2. `final_pipeline/01_data_tests`: `RUN = {"tile"}` (about 1.5 h).
3. `final_pipeline/02_long_runs`: `RUN = {"960", "1280"}` (6 h, both runs in parallel on T4 ×2).
4. `final_pipeline/03_final_models`: `ENSEMBLES = ["1280 + 960 + YOLO11s + tile"]` predicts the test frames, applies WBF and writes the CSV; `SUBMIT = True` also submits it.

Finished stages are uploaded to the Hugging Face repo, so each notebook can run in a separate session. To regenerate only the submitted CSV without training, run step 4 alone: the members' test predictions are downloaded from the repo.
