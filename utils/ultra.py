"""Ultralytics (YOLO, RT-DETR) helpers shared by notebooks 04-06.

Dataset export (pre-resized frames + YOLO labels), model configs, the loss / label-assignment changes used in the
experiments, training in a subprocess (one per GPU, so two runs can share a 2-GPU session) and a predictor that
plugs into the shared COCO evaluation of utils/common.py.

Training runs in a separate process because Ultralytics changes global state (settings, logging) and the
experiment patches below must be applied before the trainer builds its loss; a fresh process per run keeps
the notebook clean and lets each run use its own GPU.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from utils.common import (CLASSES, DATA_DIR, DEVICE, EVAL_DETS, ON_KAGGLE, OUT_DIR, PREP, RAW_DIR, ROOT, SEED,
                          load_split, run_dir, subsample)

ULTRA_VERSION = "8.4.207"                                     # tested version; the patches depend on its internals
ULTRA_DIR = OUT_DIR / "ultralytics"                           # Ultralytics run folders and downloaded weights
YOLO_DATA_DIR = Path("/tmp/yolo_data") if ON_KAGGLE else DATA_DIR / "yolo"   # Kaggle: local disk, not /kaggle/working
MODEL_CFG_DIR = ROOT / "utils" / "models"


# --------------------------------------------------------------------------------------
# Dataset export
# --------------------------------------------------------------------------------------
def _resize_one(task):
    src, dst, lbl_src, lbl_dst, long_side = task
    if not dst.exists():
        im = Image.open(src)
        s = long_side / max(im.size)
        size = (round(im.width * s), round(im.height * s))
        im.draft("RGB", size)                                    # fast JPEG decoding at reduced size
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.convert("RGB").resize(size, Image.BILINEAR).save(dst, quality=95)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    if lbl_src.exists():                                         # YOLO labels are normalised: resizing keeps them valid
        shutil.copyfile(lbl_src, lbl_dst)
    else:
        lbl_dst.write_text("")


def export_yolo(out_dir=YOLO_DATA_DIR, step=PREP["frame_step"], long_side=PREP["long_side"], max_frames=None,
                workers=4):
    """Ultralytics dataset of the subsampled train/val frames, resized once to `long_side` (same frames and
    resolution as DetDataset). Decoding the native frames (up to 3840x2160) in every epoch would make the 4 Kaggle
    CPUs the bottleneck. Labels: the provided YOLO txt files (8 classes, iscrowd=0 only; ignore regions are not
    targets). max_frames: the same reproducible subset as DetDataset (quick tests). Returns the data.yaml path."""
    out_dir = Path(out_dir)
    t0, tasks = time.time(), []
    for split in ["train", "val"]:
        frames = subsample(load_split(split)[0], step)
        if max_frames:
            frames = frames.sample(min(max_frames, len(frames)), random_state=SEED).sort_index()
        paths = []
        for f in frames.file_name:                               # images/<seq>/<frame>.jpg
            dst = out_dir / split / f
            lbl = Path(f.replace("images/", "labels/", 1)).with_suffix(".txt")
            tasks.append((RAW_DIR / split / f, dst, RAW_DIR / split / lbl, out_dir / split / lbl, long_side))
            paths.append(str(dst.resolve()))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{split}.txt").write_text("\n".join(paths))
    with get_context("fork").Pool(workers) as pool:
        for _ in pool.imap_unordered(_resize_one, tasks, chunksize=32):
            pass
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    (out_dir / "data.yaml").write_text(f"path: {out_dir.resolve()}\ntrain: train.txt\nval: val.txt\nnames:\n{names}")
    print(f"YOLO dataset: {len(tasks)} frames at {long_side} px in {out_dir} ({time.time() - t0:.0f} s)")
    return out_dir / "data.yaml"


# --------------------------------------------------------------------------------------
# Models and experiment patches
# --------------------------------------------------------------------------------------
def model_yaml(arch="yolo11s", p2=False):
    """Model config: the Ultralytics yaml of `arch`, or our P2 variant (utils/models/yolo11-p2.yaml; the scale letter
    in the file name selects the model size, as in Ultralytics)."""
    return str(MODEL_CFG_DIR / f"{arch}-p2.yaml") if p2 else f"{arch}.yaml"


def nwd(pred, target, c):
    """Normalized Wasserstein Distance similarity (Wang et al., 2021) of xyxy boxes in pixels: each box is a 2D
    Gaussian, W2^2 = |centre difference|^2 + |(w, h) difference|^2 / 4, similarity exp(-W2 / c). Unlike IoU it
    changes smoothly with a few-pixel shift of a tiny box and is non-zero for non-overlapping boxes."""
    cp, ct = (pred[:, :2] + pred[:, 2:]) / 2, (target[:, :2] + target[:, 2:]) / 2
    wp, wt = pred[:, 2:] - pred[:, :2], target[:, 2:] - target[:, :2]
    w2 = (cp - ct).pow(2).sum(-1) + ((wp - wt) / 2).pow(2).sum(-1)
    return torch.exp(-w2.clamp(min=1e-7).sqrt() / c)


def apply_patches(nwd_ratio=0.0, nwd_c=32.0):
    """Experiment change to the YOLO training loss, applied in the training process before the trainer builds it.
    nwd_ratio: box loss = (1 - r) * CIoU loss + r * NWD loss (r = 0: Ultralytics default).
    nwd_c: NWD normalising constant in input pixels (about the median box size, notebook 01)."""
    from ultralytics.utils.loss import BboxLoss

    if nwd_ratio > 0:
        orig = BboxLoss.forward

        def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                    fg_mask, imgsz, stride):
            loss_iou, loss_dfl = orig(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                      target_scores_sum, fg_mask, imgsz, stride)
            idx = fg_mask.nonzero(as_tuple=True)
            weight = target_scores[idx].sum(-1)
            s = stride[idx[1]]                                   # boxes are in grid units of their level: back to pixels
            sim = nwd(pred_bboxes[idx] * s, target_bboxes[idx] * s, nwd_c)
            loss_nwd = ((1.0 - sim) * weight).sum() / target_scores_sum
            return (1 - nwd_ratio) * loss_iou + nwd_ratio * loss_nwd, loss_dfl

        BboxLoss.forward = forward


# --------------------------------------------------------------------------------------
# Training (one subprocess per run)
# --------------------------------------------------------------------------------------
def _history(csv):
    """Ultralytics results.csv -> history in the format of common.fit (one dict per epoch).
    mAP / AP50 are Ultralytics' own validation metrics (ignore regions are NOT excluded, unlike coco_evaluate)."""
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    losses = [c for c in df.columns if c.startswith("train/")]
    return [{"epoch": int(r["epoch"]), "train_loss": float(sum(r[c] for c in losses)),
             **{c.split("/")[1]: float(r[c]) for c in losses}, "lr": float(r["lr/pg0"]),
             "mAP": float(r["metrics/mAP50-95(B)"]), "AP50": float(r["metrics/mAP50(B)"]), "time_s": float(r["time"])}
            for _, r in df.iterrows()]


def _train(cfg):
    """Body of the training subprocess. cfg: name, data, arch, p2, weights, nwd_ratio, nwd_c, train (Ultralytics
    train arguments). Writes best.pt, results.csv, args.yaml and history.json into runs/<name>/."""
    from ultralytics import YOLO
    apply_patches(cfg.get("nwd_ratio", 0.0), cfg.get("nwd_c", 32.0))
    save = ULTRA_DIR / cfg["name"]
    last = save / "weights" / "last.pt"
    if last.exists():                                            # interrupted run in this session: continue it
        try:
            YOLO(str(last)).train(resume=True)
        except AssertionError as e:                              # already finished, only the copy below is missing
            print(e)
    else:
        model = YOLO(model_yaml(cfg["arch"], cfg["p2"])).load(cfg["weights"])
        model.train(data=cfg["data"], project=str(ULTRA_DIR), name=cfg["name"], exist_ok=True,
                    device=0 if torch.cuda.is_available() else "cpu", **cfg["train"])
    d = run_dir(cfg["name"])
    for f in ["results.csv", "args.yaml"]:
        shutil.copyfile(save / f, d / f)
    json.dump(_history(save / "results.csv"), open(d / "history.json", "w"), indent=1)
    shutil.copyfile(save / "weights" / "best.pt", d / "best.pt")  # written last: marks the run as finished


def train_runs(cfgs):
    """Train the given runs (list of cfg dicts for _train), skipping finished ones (runs/<name>/best.pt exists).
    With 2+ GPUs the runs are trained concurrently, one per GPU; otherwise one after another. Output of each run
    goes to runs/<name>/train.log; a short summary is printed at the end."""
    todo = [c for c in cfgs if not (run_dir(c["name"]) / "best.pt").exists()]
    for c in cfgs:
        if c not in todo:
            print(f"{c['name']}: already trained, skipped")
    n_gpus = max(torch.cuda.device_count(), 1)
    ULTRA_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "WANDB_MODE": "disabled"}
    for i in range(0, len(todo), n_gpus):
        procs, t0 = [], time.time()
        for gpu, c in enumerate(todo[i:i + n_gpus]):
            log = open(run_dir(c["name"]) / "train.log", "w")
            gpu_env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if n_gpus > 1 else {}   # pin a GPU only when sharing them
            p = subprocess.Popen([sys.executable, "-m", "utils.ultra", json.dumps(c, default=str)], cwd=ULTRA_DIR,
                                 env={**env, **gpu_env}, stdout=log, stderr=subprocess.STDOUT)
            procs.append((c["name"], gpu, p, log))
            print(f"{c['name']}: training on GPU {gpu}, log in {run_dir(c['name']) / 'train.log'}")
        for name, gpu, p, log in procs:
            p.wait()
            log.close()
            if p.returncode:
                tail = "".join(open(run_dir(name) / "train.log").readlines()[-30:])
                raise RuntimeError(f"{name} failed (exit code {p.returncode}), end of its log:\n{tail}")
            h = json.load(open(run_dir(name) / "history.json"))
            print(f"{name}: {len(h)} epochs in {h[-1]['time_s'] / 60:.0f} min, "
                  f"Ultralytics val mAP {max(r['mAP'] for r in h):.3f} (wall time {(time.time() - t0) / 60:.0f} min)")


# --------------------------------------------------------------------------------------
# Inference for the shared COCO evaluation
# --------------------------------------------------------------------------------------
def load_ultra(path):
    """Trained Ultralytics model (YOLO or RT-DETR) from a best.pt."""
    from ultralytics import RTDETR, YOLO
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    is_detr = "RTDETR" in type(ckpt["model"]).__name__
    return (RTDETR if is_detr else YOLO)(str(path))


def ultra_predictor(model, imgsz=PREP["long_side"], nms=0.7, half=True):
    """predict_fn for common.predict_dataset: DetDataset images (float RGB in [0, 1], already resized to imgsz on the
    long side) -> boxes in the same pixels, scores and 0-based labels (use label_offset=0). Ultralytics only pads the
    short side to a multiple of the stride. nms: IoU of the final NMS (Ultralytics default 0.7)."""
    dev = 0 if DEVICE.type == "cuda" else "cpu"

    def predict(images):
        arrs = [np.ascontiguousarray((im.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)[:, :, ::-1])
                for im in images]                                # Ultralytics expects BGR uint8 (OpenCV convention)
        res = model.predict(arrs, imgsz=imgsz, conf=0.001, iou=nms, max_det=EVAL_DETS, half=half and dev == 0,
                            device=dev, verbose=False)
        return [dict(boxes=r.boxes.xyxy, scores=r.boxes.conf, labels=r.boxes.cls.long()) for r in res]
    return predict


if __name__ == "__main__":                                       # python -m utils.ultra '<cfg json>'
    _train(json.loads(sys.argv[1]))
