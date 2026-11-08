"""Shared code for the HW2 notebooks (VisDrone multiple object video detection).

Config and paths (local or Kaggle), data access, a framework-agnostic dataset, COCO evaluation,
a training loop, run saving/loading and plotting helpers. Every notebook does `from common import *`.
"""

import contextlib
import functools
import io
import json
import platform
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageDraw
from tqdm import tqdm


def _check_versions(min_torch=(2, 3), min_torchvision=(0, 18)):
    """Fail early with a clear message if the installed torch/torchvision is older than the code needs."""
    ver = lambda v: tuple(int(x) for x in v.split("+")[0].split(".")[:2])
    for lib, minimum in [(torch, min_torch), (torchvision, min_torchvision)]:
        if ver(lib.__version__) < minimum:
            raise ImportError(f"{lib.__name__} {lib.__version__} is too old, need >= {'.'.join(map(str, minimum))} "
                              f"(pip install -r requirements.txt)")


_check_versions()

from pycocotools.cocoeval import COCOeval          # noqa: E402  (imported after the version check)
from torchvision import tv_tensors                 # noqa: E402
from torchvision.transforms import v2              # noqa: E402

# --------------------------------------------------------------------------------------
# Config and paths
# --------------------------------------------------------------------------------------
SEED = 42
COMPETITION = "multiple-object-video-detection-hw-2-ucu-ai-2026"
HF_REPO = "BohdanZasymovych/ai-hw2-object-detection"            # Hugging Face model repo for runs and checkpoints
ON_KAGGLE = Path("/kaggle/working").exists()


def _find_raw_dir():
    """Kaggle: the attached competition folder is the one that contains test_images.csv."""
    hits = sorted(Path("/kaggle/input").glob("**/test_images.csv"))
    return hits[0].parent if hits else Path("/kaggle/input") / COMPETITION


ROOT = Path(__file__).resolve().parents[1]                                    # repository root (utils/ is one level below)
DATA_DIR = ROOT / "data"
RAW_DIR = _find_raw_dir() if ON_KAGGLE else DATA_DIR / "raw"                  # read-only on Kaggle
OUT_DIR = Path("/kaggle/working") if ON_KAGGLE else ROOT                      # where outputs are written
RUNS_DIR = OUT_DIR / "runs"                                                   # one folder per experiment
SUBMISSIONS_DIR = OUT_DIR / "submissions"                                     # competition CSVs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]   # COCO category_id = index + 1
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]

pd.set_option("display.width", 140)
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})


def seed_everything(seed=SEED):
    """Seed python, numpy and torch (CPU and GPU) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_data():
    """Download and extract the competition data into data/raw (no-op if present or on Kaggle)."""
    if ON_KAGGLE or (RAW_DIR.exists() and any(RAW_DIR.iterdir())):
        print(f"Using dataset in {RAW_DIR}")
        return
    import zipfile
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / f"{COMPETITION}.zip"
    if not archive.exists():
        subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)], check=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW_DIR)
    print("Extracted to", RAW_DIR)


# --------------------------------------------------------------------------------------
# Data access and protocol
# --------------------------------------------------------------------------------------
# Preprocessing / training protocol shared by all detectors (values justified in section 2)
PREP = dict(
    frame_step=3,                # keep every 3rd frame of each sequence (train and val)
    long_side=1280,              # resize the long side, keep aspect ratio
    train_aug=dict(hflip=0.5, scale_jitter=(0.75, 1.25), color_jitter=0.2),   # box-aware, train only
    mask_ignore=False,           # paint ignore regions grey in the image (off by default)
    min_box_px=2,                # drop boxes smaller than this after augmentation
    seed=SEED,
)
EVAL_DETS = 300                  # max detections per image kept for evaluation (up to 180 objects per frame)


@functools.lru_cache(maxsize=None)
def load_split(split):
    """Return (frames, boxes, ignore) DataFrames of a split, in native pixels with xywh boxes. Treat as read-only.
    boxes: iscrowd=0 targets; ignore: ignore regions (iscrowd=1, orig_category_id 0 or 11), deduplicated over the per-class copies."""
    d = json.load(open(RAW_DIR / split / "annotations_coco.json"))
    frames = pd.DataFrame(d["images"]).rename(columns={"id": "image_id", "video": "seq"})
    anns = pd.DataFrame(d["annotations"])
    anns[["x", "y", "w", "h"]] = pd.DataFrame(anns["bbox"].tolist(), index=anns.index)
    anns = anns.merge(frames[["image_id", "seq", "width", "height"]], on="image_id").drop(columns="bbox")
    boxes = anns[anns.iscrowd == 0].copy()
    ignore = anns[anns.iscrowd == 1].drop_duplicates(["image_id", "x", "y", "w", "h"]).copy()
    boxes["cls"] = boxes.category_id.map(lambda c: CLASSES[c - 1])
    boxes["rel_w"], boxes["rel_h"] = boxes.w / boxes.width, boxes.h / boxes.height
    boxes["size"] = (boxes.w * boxes.h) ** 0.5               # sqrt(area) in native pixels
    return frames, boxes, ignore


def subsample(df, step=PREP["frame_step"]):
    """Deterministic frame subsampling per sequence (frames are numbered from 1)."""
    return df[(df.frame_id - 1) % step == 0]


@functools.lru_cache(maxsize=None)
def coco_gt(split, step=PREP["frame_step"]):
    """pycocotools ground truth of the subsampled frames; iscrowd=1 regions are kept so evaluation ignores them."""
    from pycocotools.coco import COCO
    d = json.load(open(RAW_DIR / split / "annotations_coco.json"))
    keep = {i["id"] for i in d["images"] if (i["frame_id"] - 1) % step == 0}
    gt = COCO()
    gt.dataset = dict(images=[i for i in d["images"] if i["id"] in keep],
                      annotations=[a for a in d["annotations"] if a["image_id"] in keep], categories=d["categories"])
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def export_coco(split, path, step=PREP["frame_step"]):
    """Write the subsampled split as a COCO json (for MMDetection, Hugging Face, ...)."""
    json.dump(coco_gt(split, step).dataset, open(path, "w"))
    return path


# --------------------------------------------------------------------------------------
# Dataset and transforms
# --------------------------------------------------------------------------------------
class DetDataset(torch.utils.data.Dataset):
    """Frames of split 'train' | 'val' | 'test' -> (image float32 CxHxW in [0,1], target dict).
    target: boxes (N,4) xyxy and ignore (M,4) in resized pixels, labels (N,), image_id, scale=(sx, sy) and
    orig_size=(h, w); divide resized coordinates by scale to get native pixels. Test frames have no boxes."""

    def __init__(self, split, train=False, step=None, long_side=PREP["long_side"], aug=PREP["train_aug"],
                 mask_ignore=PREP["mask_ignore"], label_offset=1, max_frames=None):
        self.split, self.train, self.long_side = split, train, long_side
        self.aug, self.mask_ignore, self.label_offset = (aug if train else None), mask_ignore, label_offset
        if split == "test":
            frames = pd.read_csv(RAW_DIR / "test_images.csv")
            frames["path"] = [RAW_DIR / f for f in frames.file_name]
            self.boxes, self.ignore = {}, {}
        else:
            frames, boxes, ignore = load_split(split)
            frames = subsample(frames, PREP["frame_step"] if step is None else step).copy()
            frames["path"] = [RAW_DIR / split / f for f in frames.file_name]
            ids = set(frames.image_id)
            if max_frames:                                        # reproducible random subset (quick experiments)
                frames = frames.sample(min(max_frames, len(frames)), random_state=SEED).sort_index()
                ids = set(frames.image_id)
            self.boxes = {i: (g[["x", "y", "w", "h"]].to_numpy(np.float32), g.category_id.to_numpy())
                          for i, g in boxes[boxes.image_id.isin(ids)].groupby("image_id")}
            self.ignore = {i: g[["x", "y", "w", "h"]].to_numpy(np.float32)
                           for i, g in ignore[ignore.image_id.isin(ids)].groupby("image_id")}
        self.rows = frames[["image_id", "path"]].to_dict("records")
        self.image_ids = [r["image_id"] for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def _train_tf(self, h, w):
        a, ts = self.aug, []
        if a.get("hflip"):
            ts.append(v2.RandomHorizontalFlip(a["hflip"]))
        if a.get("scale_jitter"):                                 # zoom, then crop/pad back to the original canvas
            ts += [v2.ScaleJitter((h, w), scale_range=a["scale_jitter"]), v2.RandomCrop((h, w), pad_if_needed=True)]
        if a.get("color_jitter"):
            ts.append(v2.ColorJitter(a["color_jitter"], a["color_jitter"], a["color_jitter"]))
        ts.append(v2.SanitizeBoundingBoxes(min_size=PREP["min_box_px"]))
        return v2.Compose(ts)

    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(r["path"])
        ow, oh = im.size
        s = self.long_side / max(ow, oh)
        nw, nh = round(ow * s), round(oh * s)
        im.draft("RGB", (nw, nh))                                 # fast JPEG decoding at reduced size
        im = im.convert("RGB").resize((nw, nh), Image.BILINEAR)
        sx, sy = nw / ow, nh / oh
        xywh, cats = self.boxes.get(r["image_id"], (np.zeros((0, 4), np.float32), np.zeros(0, np.int64)))
        boxes = np.stack([xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]], 1) * [sx, sy, sx, sy]
        ign = self.ignore.get(r["image_id"], np.zeros((0, 4), np.float32))
        ign = np.stack([ign[:, 0], ign[:, 1], ign[:, 0] + ign[:, 2], ign[:, 1] + ign[:, 3]], 1) * [sx, sy, sx, sy]
        if self.mask_ignore:
            d = ImageDraw.Draw(im)
            for b in ign:
                d.rectangle(b.tolist(), fill=(114, 114, 114))
        img = tv_tensors.Image(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
        tgt = {"boxes": tv_tensors.BoundingBoxes(torch.as_tensor(boxes, dtype=torch.float32), format="XYXY", canvas_size=(nh, nw)),
               "labels": torch.as_tensor(cats - 1 + self.label_offset, dtype=torch.int64)}
        if self.aug:
            img, tgt = self._train_tf(nh, nw)(img, tgt)
        tgt = {"boxes": tgt["boxes"].as_subclass(torch.Tensor), "labels": tgt["labels"], "image_id": r["image_id"],
               "scale": (sx, sy), "orig_size": (oh, ow), "ignore": torch.as_tensor(ign, dtype=torch.float32)}
        return img.as_subclass(torch.Tensor).float() / 255, tgt


def collate(batch):
    return tuple(zip(*batch))


def _seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


def make_loader(ds, batch_size=4, shuffle=None, num_workers=4):
    """DataLoader with list-style batches (images may differ in size) and reproducible shuffling/augmentation.
    Uses fork workers (Python 3.14 defaults to forkserver, which cannot pickle classes defined in a notebook)."""
    g = torch.Generator().manual_seed(SEED)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=ds.train if shuffle is None else shuffle, num_workers=num_workers,
        collate_fn=collate, generator=g, pin_memory=DEVICE.type == "cuda", persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker, multiprocessing_context="fork" if num_workers > 0 else None)


def tv_size_kwargs(long_side=PREP["long_side"]):
    """min_size/max_size for torchvision detectors so they keep our resolution (16:9) instead of resizing to 800/1333."""
    return dict(min_size=round(long_side * 9 / 16), max_size=long_side)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def predict_dataset(predict_fn, ds, batch_size=4, num_workers=4, score_thr=0.001, max_dets=EVAL_DETS):
    """Run predict_fn over a dataset -> (COCO-format results in native pixels, inference ms per image).
    Timing covers predict_fn only (not data loading) and skips the first batch as warm-up."""
    results, times = [], []
    for images, targets in tqdm(make_loader(ds, batch_size, False, num_workers), leave=False, mininterval=10):
        _sync(); t0 = time.perf_counter()
        outs = predict_fn(list(images))
        _sync(); times.append((time.perf_counter() - t0, len(images)))
        for out, tg in zip(outs, targets):
            b = out["boxes"].float().cpu().numpy() / np.array([*tg["scale"], *tg["scale"]], dtype=np.float32)
            sc, lb = out["scores"].float().cpu().numpy(), out["labels"].cpu().numpy()
            k = np.where(sc >= score_thr)[0]
            k = k[np.argsort(-sc[k])][:max_dets]
            results += [dict(image_id=tg["image_id"], category_id=int(lb[j] - ds.label_offset + 1), score=float(sc[j]),
                             bbox=[round(float(b[j, 0]), 2), round(float(b[j, 1]), 2),
                                   round(float(b[j, 2] - b[j, 0]), 2), round(float(b[j, 3] - b[j, 1]), 2)]) for j in k]
    timed = times[1:] or times
    return results, 1000 * sum(t for t, _ in timed) / max(1, sum(n for _, n in timed))


def coco_evaluate(results, split="val", step=None, image_ids=None, max_dets=EVAL_DETS):
    """COCO metrics for results in COCO format (image_id, category_id, bbox xywh native px, score)."""
    gt = coco_gt(split, PREP["frame_step"] if step is None else step)
    ids = list(gt.getImgIds() if image_ids is None else image_ids)
    if not results:
        return {**dict.fromkeys(["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR"], 0.0), "per_class": {}}
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(results), "bbox")
        E.params.imgIds, E.params.maxDets = ids, [1, 10, max_dets]
        E.evaluate(); E.accumulate()
    # Computed from the raw arrays: E.summarize() hard-codes maxDets=100 for mAP, which breaks with max_dets=300.
    P, R = E.eval["precision"], E.eval["recall"]                   # P: IoU x recall x class x area x maxDets
    mean = lambda a: float(a[a > -1].mean()) if (a > -1).any() else float("nan")
    m = dict(mAP=mean(P[:, :, :, 0, 2]), AP50=mean(P[0, :, :, 0, 2]), AP75=mean(P[5, :, :, 0, 2]),
             AP_small=mean(P[:, :, :, 1, 2]), AP_medium=mean(P[:, :, :, 2, 2]), AP_large=mean(P[:, :, :, 3, 2]),
             AR=mean(R[:, :, 0, 2]))
    m["per_class"] = {c: mean(P[:, :, k, 0, 2]) for k, c in enumerate(CLASSES)}
    return m


def evaluate_model(predict_fn, split="val", max_frames=None, batch_size=4, num_workers=4, label_offset=1,
                   long_side=PREP["long_side"]):
    """Full evaluation of a model on a split -> metrics dict (+ ms_per_img). long_side: input resolution of the frames."""
    ds = DetDataset(split, label_offset=label_offset, max_frames=max_frames, long_side=long_side)
    results, ms = predict_dataset(predict_fn, ds, batch_size, num_workers)
    return {**coco_evaluate(results, split, image_ids=ds.image_ids), "ms_per_img": ms}


def torchvision_predictor(model, device=DEVICE, amp=True):
    """predict_fn for torchvision detectors (Faster R-CNN, RetinaNet, FCOS, SSD).
    Build the model with box_score_thresh/score_thresh=0.001 and detections_per_img=EVAL_DETS for evaluation."""
    def predict(images):
        model.eval()
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            return model([im.to(device) for im in images])
    return predict


# --------------------------------------------------------------------------------------
# Training, tracking and saving runs
# --------------------------------------------------------------------------------------
def _save_atomic(obj, path):
    """torch.save through a temp file, so a disconnect during the write cannot leave a corrupt checkpoint."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _optimizer_step(model, optimizer, scaler, grad_clip):
    if grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True)


def fit(model, train_loader, optimizer, epochs, scheduler=None, eval_fn=None, amp=True, grad_clip=10.0, accum_steps=1,
        ckpt_path=None, last_path=None, resume_from=None, logger=None, time_budget_s=None, on_epoch_end=None):
    """Train a torchvision-style detector. eval_fn() -> metrics dict (with 'mAP') runs after every epoch.
    ckpt_path: best weights by val mAP (model only). last_path: full training state written after every epoch
    (model, optimizer, scheduler, AMP scaler, RNG states, history); resume_from: such a file, training continues
    with the next epoch (build the model, optimizer and scheduler exactly as in the first run).
    on_epoch_end(epoch, row) is called after each epoch (after the checkpoints are written), e.g. to push them to the Hub.
    accum_steps: gradients of this many batches are summed before one optimizer step (effective batch = batch size x accum_steps),
    for GPUs too small for the full batch; equivalent to the full batch for torchvision detectors (their backbone BatchNorm is frozen).
    Returns history: one dict per epoch with mean losses, validation metrics and elapsed time."""
    model.to(DEVICE)
    use_amp = amp and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    history, best, elapsed, start_ep = [], -1.0, 0.0, 1
    if resume_from and Path(resume_from).exists():
        st = torch.load(resume_from, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["optimizer"])
        if scheduler and st["scheduler"]:
            scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        history, best, elapsed, start_ep = st["history"], st["best"], st["elapsed"], st["epoch"] + 1
        torch.set_rng_state(st["rng_cpu"].cpu())
        if torch.cuda.is_available() and st["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in st["rng_cuda"]])
        if train_loader.generator is not None and st["loader_rng"] is not None:
            train_loader.generator.set_state(st["loader_rng"].cpu())
        print(f"Resumed from {resume_from} at epoch {start_ep}/{epochs}")
    t0 = time.time() - elapsed                                    # elapsed time continues across resumes
    for ep in range(start_ep, epochs + 1):
        model.train()
        sums, n, pending = defaultdict(float), 0, 0
        optimizer.zero_grad(set_to_none=True)
        for images, targets in tqdm(train_loader, desc=f"epoch {ep}/{epochs}", leave=False, mininterval=10):
            images = [im.to(DEVICE, non_blocking=True) for im in images]
            targets = [{k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            with torch.autocast(DEVICE.type, enabled=use_amp):
                losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):                          # skip a diverged batch (and its accumulation group)
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                continue
            scaler.scale(loss / accum_steps).backward()
            pending += 1
            if pending == accum_steps:
                _optimizer_step(model, optimizer, scaler, grad_clip)
                pending = 0
            n += 1
            sums["train_loss"] += loss.item()
            for k, v in losses.items():
                sums[k] += v.item()
        if pending:                                               # leftover batches of an incomplete accumulation group
            _optimizer_step(model, optimizer, scaler, grad_clip)
        row = {"epoch": ep, **{k: v / max(n, 1) for k, v in sums.items()}, "lr": optimizer.param_groups[0]["lr"]}  # lr used in this epoch
        if scheduler:
            scheduler.step()
        if eval_fn:
            row.update({k: v for k, v in eval_fn().items() if not isinstance(v, dict)})
            if ckpt_path and row["mAP"] > best:
                best = row["mAP"]
                _save_atomic(model.state_dict(), ckpt_path)
            elif row["mAP"] > best:
                best = row["mAP"]
        row["time_s"] = time.time() - t0
        history.append(row)
        if last_path:                                             # full state after every epoch, for resuming
            _save_atomic(dict(
                model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                scheduler=scheduler.state_dict() if scheduler else None, epoch=ep, history=history, best=best,
                elapsed=row["time_s"], rng_cpu=torch.get_rng_state(),
                rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                loader_rng=train_loader.generator.get_state() if train_loader.generator is not None else None), last_path)
        if logger:
            logger.log(row, step=ep)
        if on_epoch_end:
            on_epoch_end(ep, row)
        if time_budget_s and row["time_s"] > time_budget_s:
            break
    return history


def plot_history(histories, metric="mAP"):
    """Compare runs: validation metric against epoch and against wall-clock time. histories = {name: history}."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.5))
    for name, h in histories.items():
        axs[0].plot([r["epoch"] for r in h], [r[metric] for r in h], marker="o", label=name)
        axs[1].plot([r["time_s"] / 60 for r in h], [r[metric] for r in h], marker="o", label=name)
    axs[0].set_xlabel("epoch"); axs[1].set_xlabel("minutes")
    for ax in axs:
        ax.set_ylabel(f"val {metric}")
    axs[0].legend(); plt.tight_layout(); plt.show()


class RunLogger:
    """Thin Weights & Biases wrapper; does nothing if disabled or wandb is not installed."""

    def __init__(self, name=None, config=None, project="hw2-visdrone", enabled=False):
        self.run = None
        if enabled:
            try:
                import wandb
                self.run = wandb.init(project=project, name=name, config=config)
            except ImportError:
                print("wandb is not installed, logging disabled")

    def log(self, data, step=None):
        if self.run:
            self.run.log(data, step=step)

    def finish(self):
        if self.run:
            self.run.finish()


def env_info():
    """Versions and hardware of the current run (stored with every saved run)."""
    import torchvision
    return dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", on_kaggle=ON_KAGGLE)


def run_dir(name):
    """Folder of an experiment (created on demand): runs/<name>/."""
    d = RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(name, config=None, history=None, metrics=None, preds=None, model=None):
    """Save everything about one experiment into runs/<name>/ so it can be downloaded from Kaggle and analysed locally:
    config.json (+ environment), history.json (per epoch), metrics.json, preds_val.json (COCO results), best.pt (weights)."""
    d = run_dir(name)
    json.dump({**(config or {}), "env": env_info()}, open(d / "config.json", "w"), indent=2, default=str)
    if history is not None:
        json.dump(history, open(d / "history.json", "w"), indent=1)
    if metrics is not None:
        json.dump(metrics, open(d / "metrics.json", "w"), indent=1)
    if preds is not None:
        json.dump(preds, open(d / "preds_val.json", "w"))
    if model is not None:
        torch.save(model.state_dict(), d / "best.pt")
    return d


def fetch_run(name, weights=False, hint=""):
    """Local folder of a finished run; downloaded from the HF repo if it was produced in another session.
    hint: appended to the error message (e.g. which RUN stage computes it)."""
    d = RUNS_DIR / name
    if not (d / "metrics.json").exists() or (weights and not (d / "best.pt").exists()):
        pull_run(name, weights=weights)
    if not (d / "metrics.json").exists():
        raise FileNotFoundError(f"Run '{name}' has no results locally or in the HF repo {HF_REPO}. {hint}".strip())
    return d


def load_results(runs_dir=RUNS_DIR):
    """Comparison table of all saved runs (metrics.json of every runs/<name>/), one row per experiment."""
    rows = {}
    for f in sorted(Path(runs_dir).glob("*/metrics.json")):
        m = json.load(open(f))
        rows[f.parent.name] = {k: v for k, v in m.items() if k != "per_class"}
    return pd.DataFrame(rows).T.round(3)


# --------------------------------------------------------------------------------------
# Hugging Face Hub: move runs between Kaggle and the laptop
# --------------------------------------------------------------------------------------
def hf_token():
    """Token from the HF_TOKEN env var or the Kaggle secret named HF_TOKEN; None falls back to a cached `hf auth login`."""
    import os
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    if ON_KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as e:
            print(f"Kaggle secret HF_TOKEN is not readable ({type(e).__name__}: {e}). Add it in Add-ons -> Secrets "
                  "and tick its checkbox to attach it to this notebook.")
            return None
    return None


def push_run(name, repo_id=HF_REPO, private=False, weights=True, last=False):
    """Upload runs/<name>/ to the model repo `repo_id` (default HF_REPO, created if missing) under runs/<name>/.
    weights=False skips best.pt; last=True also uploads last.pt (the full state, needed to resume elsewhere)."""
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())
    api.create_repo(repo_id, private=private, exist_ok=True)
    ignore = ([] if weights else ["best.pt"]) + ([] if last else ["last.pt"]) + ["*.tmp"]
    api.upload_folder(folder_path=str(run_dir(name)), path_in_repo=f"runs/{name}", repo_id=repo_id,
                      ignore_patterns=ignore, commit_message=f"run {name}")
    print(f"Uploaded runs/{name} -> https://huggingface.co/{repo_id}/tree/main/runs/{name}")


def pull_run(name, repo_id=HF_REPO, weights=True, last=False):
    """Download runs/<name>/ from the Hub into the local runs/ folder (metrics and predictions always; weights optional)."""
    from huggingface_hub import snapshot_download
    ignore = ([] if weights else ["*/best.pt"]) + ([] if last else ["*/last.pt"])
    snapshot_download(repo_id, allow_patterns=[f"runs/{name}/*"], ignore_patterns=ignore,
                      local_dir=str(OUT_DIR), token=hf_token())
    return RUNS_DIR / name


def check_hf_write(repo_id=HF_REPO):
    """Fail early (before any training) if results cannot be uploaded to `repo_id`.
    Real write test: uploads a constant marker file; the Hub skips identical re-uploads, so repeated checks add no commits."""
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())
    try:
        user = api.whoami()["name"]
        api.create_repo(repo_id, exist_ok=True)
        api.upload_file(path_or_fileobj=b"write access check\n", path_in_repo=".write_check", repo_id=repo_id,
                        commit_message="write access check")
    except Exception as e:
        raise RuntimeError(f"No write access to the Hugging Face repo {repo_id} ({type(e).__name__}: {e}). "
                           "Set HF_TOKEN to a write token (Kaggle: Add-ons -> Secrets; locally: `hf auth login`).") from e
    print(f"Hugging Face write access OK: {user} -> {repo_id}")


def push_file(path, repo_id=HF_REPO, path_in_repo=None, private=False):
    """Upload a single file (e.g. last.pt every few epochs from `fit(on_epoch_end=...)`)."""
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo or Path(path).name, repo_id=repo_id)


# --------------------------------------------------------------------------------------
# Test predictions and Kaggle submission
# --------------------------------------------------------------------------------------
def write_submission(results, image_ids, path):
    """COCO-format results (native px, 1-based category_id) -> competition CSV: one row per image, 'class score x y w h'
    groups with 0-based classes and the top-left corner in pixels; images without detections stay empty."""
    rows = {i: [] for i in image_ids}
    for r in results:
        x, y, w, h = r["bbox"]
        rows[r["image_id"]].append(f"{r['category_id'] - 1} {r['score']:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}")
    pd.DataFrame({"image_id": list(rows), "PredictionString": [" ".join(v) for v in rows.values()]}).to_csv(path, index=False)
    return Path(path)


def check_submission(path, max_dets=500, tol=1.0):
    """Validate a submission CSV against test_images.csv before uploading it; raises ValueError on the first problem.
    Returns (frames, boxes). tol: allowed overshoot of a box beyond the image, in pixels."""
    sub = pd.read_csv(path, keep_default_na=False, dtype=str)
    test = pd.read_csv(RAW_DIR / "test_images.csv")
    if list(sub.columns) != ["image_id", "PredictionString"]:
        raise ValueError(f"wrong columns: {list(sub.columns)}")
    if len(sub) != len(test) or set(sub.image_id) != set(test.image_id):
        raise ValueError(f"image ids differ from test_images.csv ({len(sub)} rows, expected {len(test)})")
    sizes = test.set_index("image_id")[["width", "height"]]
    boxes = 0
    for i, s in zip(sub.image_id, sub.PredictionString):
        a = np.array(s.split(), dtype=np.float64)
        if a.size % 6:
            raise ValueError(f"{i}: the prediction string is not made of 6-value groups")
        a = a.reshape(-1, 6)
        W, H = sizes.loc[i]
        if len(a) > max_dets:
            raise ValueError(f"{i}: {len(a)} detections, at most {max_dets} are scored")
        if ((a[:, 0] < 0) | (a[:, 0] > len(CLASSES) - 1) | (a[:, 0] % 1 != 0)).any():
            raise ValueError(f"{i}: class id outside 0-{len(CLASSES) - 1}")
        if ((a[:, 1] < 0) | (a[:, 1] > 1)).any():
            raise ValueError(f"{i}: score outside [0, 1]")
        if ((a[:, 2] < -tol) | (a[:, 3] < -tol) | (a[:, 2] + a[:, 4] > W + tol) | (a[:, 3] + a[:, 5] > H + tol)
                | (a[:, 4] < 0) | (a[:, 5] < 0)).any():
            raise ValueError(f"{i}: box outside the {W}x{H} image")
        boxes += len(a)
    return len(sub), boxes


def make_submission(predict_fn, name, long_side=PREP["long_side"], label_offset=1, batch_size=1, num_workers=4):
    """Run predict_fn (as for predict_dataset) on all test frames, write the competition CSV to
    submissions/submission_<name>.csv and validate it. Returns the path."""
    ds = DetDataset("test", label_offset=label_offset, long_side=long_side)
    results, ms = predict_dataset(predict_fn, ds, batch_size, num_workers)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = write_submission(results, ds.image_ids, SUBMISSIONS_DIR / f"submission_{name}.csv")
    frames, boxes = check_submission(path)
    print(f"{path}: {frames} frames, {boxes} boxes, {ms:.0f} ms/img, {path.stat().st_size / 1e6:.0f} MB")
    return path


def _kaggle_cli(*args):
    """Run the Kaggle CLI of this Python environment -> stdout. On Kaggle the credentials come from the secrets
    KAGGLE_USERNAME and KAGGLE_KEY; locally the CLI uses its own configuration (~/.kaggle)."""
    import os
    import shutil
    import sys
    env = dict(os.environ)
    if ON_KAGGLE:
        env.update(kaggle_secrets())
    local = Path(sys.executable).parent / "kaggle"
    exe = str(local) if local.exists() else shutil.which("kaggle") or "kaggle"
    r = subprocess.run([exe, "competitions", *args], env=env, capture_output=True, text=True)
    if r.returncode:                                              # show the CLI's own message, not only its exit code
        raise RuntimeError(f"kaggle competitions {args[0]} failed (exit code {r.returncode}):\n{r.stdout}\n{r.stderr}".strip())
    return r.stdout


def kaggle_secrets():
    """KAGGLE_USERNAME and KAGGLE_KEY from the Kaggle secrets; call early to fail before a long run."""
    from kaggle_secrets import UserSecretsClient
    try:
        return {k: UserSecretsClient().get_secret(k) for k in ["KAGGLE_USERNAME", "KAGGLE_KEY"]}
    except Exception as e:
        raise RuntimeError("Add the secrets KAGGLE_USERNAME and KAGGLE_KEY (Add-ons -> Secrets, from a Kaggle API token) "
                           f"and attach them to this notebook ({type(e).__name__}: {e})") from e


def _submission_row(table, message):
    """Newest row of the `submissions --csv` output with this description, or None."""
    if "publicScore" not in table:                                # "No submissions found"
        return None
    df = pd.read_csv(io.StringIO(table[table.index("ref"):]), dtype=str, keep_default_na=False)
    rows = df[df.description == message]
    return rows.iloc[0].to_dict() if len(rows) else None


def submit_to_kaggle(path, message, wait_s=600, poll_s=20):
    """Upload a validated submission CSV (uses one of the daily submissions) and wait for its public score.
    Returns the row of the submissions table (status, publicScore, ...), or None if it is still being scored."""
    check_submission(path)
    _kaggle_cli("submit", COMPETITION, "-f", str(path), "-m", message)
    print(f"Submitted {path}: {message}")
    t0 = time.time()
    while time.time() - t0 < wait_s:
        time.sleep(poll_s)
        row = _submission_row(_kaggle_cli("submissions", COMPETITION, "--csv"), message)
        if row and "pending" not in row["status"].lower():
            print(f"status: {row['status']} | public score: {row['publicScore']}")
            return row
    print(f"Still being scored after {wait_s} s; see the Submissions tab of the competition")
    return None


# --------------------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------------------
def draw_frame(split, image_id, ax, preds=None, score_thr=0.3, show_gt=True, color_preds=False):
    """Draw GT boxes (coloured by class), ignore regions (red shade) and optional predictions on a native-resolution frame.
    preds: COCO-format results list (xywh native px) containing this image_id; drawn white, or coloured by predicted class if color_preds.
    show_gt=False hides the GT boxes (ignore regions stay)."""
    frames, boxes, ignore = load_split(split)
    row = frames.set_index("image_id").loc[image_id]
    im = Image.open(RAW_DIR / split / row.file_name).convert("RGBA")
    ov = Image.new("RGBA", im.size)
    od = ImageDraw.Draw(ov)
    for r in ignore[ignore.image_id == image_id].itertuples():
        od.rectangle([r.x, r.y, r.x + r.w, r.y + r.h], fill=(255, 0, 0, 90))
    im = Image.alpha_composite(im, ov).convert("RGB")
    d, lw = ImageDraw.Draw(im), max(2, im.width // 700)
    if show_gt:
        for r in boxes[boxes.image_id == image_id].itertuples():
            d.rectangle([r.x, r.y, r.x + r.w, r.y + r.h], outline=COLORS[r.category_id - 1], width=lw)
    for p in preds or []:
        if p["image_id"] == image_id and p["score"] >= score_thr:
            x, y, w, h = p["bbox"]
            d.rectangle([x, y, x + w, y + h], outline=COLORS[p["category_id"] - 1] if color_preds else "white", width=max(1, lw - 1))
    ax.imshow(im); ax.axis("off"); ax.set_title(f"{row.seq} #{row.frame_id}", fontsize=9)


# --------------------------------------------------------------------------------------
# Anchor fit and error analysis (shared by the detector notebooks)
# --------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _train_wh(long_side=PREP["long_side"], n=100_000):
    """(w, h) of a random sample of train boxes at the input resolution, and a mask of the COCO-small ones."""
    _, bx, _ = load_split("train")
    wh = bx[["w", "h"]].to_numpy() * (long_side / bx[["width", "height"]].max(axis=1).to_numpy())[:, None]
    idx = np.random.default_rng(SEED).choice(len(wh), n, replace=False)
    return wh[idx], (bx["size"].to_numpy() < 32)[idx]


def _best_anchor_iou(wh, sizes, ratios):
    """Best IoU of each (w, h) box with any anchor shape (position ignored)."""
    sizes = [s for lvl in sizes for s in (lvl if isinstance(lvl, (tuple, list)) else (lvl,))]
    a = np.array([(s / np.sqrt(r), s * np.sqrt(r)) for s in sizes for r in ratios])        # (w, h), r = h / w
    inter = np.minimum(wh[:, None, 0], a[None, :, 0]) * np.minimum(wh[:, None, 1], a[None, :, 1])
    return (inter / (wh[:, 0:1] * wh[:, 1:2] + a[:, 0] * a[:, 1] - inter)).max(1)


def anchor_fit(sizes, ratios=(0.5, 1.0, 2.0), long_side=PREP["long_side"]):
    """How well a set of anchor shapes fits the train boxes: best IoU of each box with any anchor shape (position ignored).
    sizes: anchor sizes in input pixels, flat or one tuple per level."""
    wh, small = _train_wh(long_side)
    best = _best_anchor_iou(wh, sizes, ratios)
    return {"median best IoU": np.median(best), "% IoU>=0.5": (best >= .5).mean() * 100, "% IoU>=0.7": (best >= .7).mean() * 100,
            "% IoU>=0.7 (small)": (best[small] >= .7).mean() * 100}


def anchor_fit_by_size(configs, edges=(32, 96, 150), ratios=(0.5, 1.0, 2.0), long_side=PREP["long_side"]):
    """anchor_fit split by box size sqrt(w*h) at the input resolution. configs: {name: sizes} -> DataFrame,
    rows = size buckets + all boxes (with their share of boxes), columns = (config, median best IoU | % IoU>=0.7)."""
    wh, _ = _train_wh(long_side)
    size = np.sqrt(wh[:, 0] * wh[:, 1])
    bins = np.digitize(size, edges)
    labels = [f"<{edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}"]
    out = {}
    for name, sizes in configs.items():
        best = _best_anchor_iou(wh, sizes, ratios)
        for i, lab in enumerate(labels + ["all"]):
            m = bins == i if lab != "all" else np.ones(len(best), bool)
            out[lab] = {**out.get(lab, {}), ("", "% of boxes"): m.mean() * 100,
                        (name, "median best IoU"): np.median(best[m]), (name, "% IoU>=0.7"): (best[m] >= .7).mean() * 100}
    df = pd.DataFrame(out).T
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.name = "box size (px)"
    return df


def match_errors(preds, image_ids, split="val", score_thr=0.3, iou_thr=0.5):
    """Error types at a fixed score threshold -> (gt, det) DataFrames.
    GT boxes: correct | crowded (overlapped only by a same-class box already matched to a neighbour) |
              wrong_class (overlapped only by another class) | mislocalized (best IoU 0.1-0.5) | missed.
    Detections: tp | duplicate | wrong_class | ignored (centre in an ignore region) | background."""
    from torchvision.ops import box_iou
    _, boxes, ignore = load_split(split)
    P = pd.DataFrame(preds)
    P = P[P.score >= score_thr]
    bg, ig, pg = boxes.groupby("image_id"), ignore.groupby("image_id"), P.groupby("image_id")
    xyxy = lambda a: torch.tensor(np.c_[a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]], dtype=torch.float32)
    gts, dets = [], []
    for i in image_ids:
        g = bg.get_group(i) if i in bg.groups else boxes.iloc[:0]
        p = pg.get_group(i).sort_values("score", ascending=False) if i in pg.groups else P.iloc[:0]
        G = xyxy(g[["x", "y", "w", "h"]].to_numpy(np.float32))
        D = xyxy(np.array(p.bbox.tolist(), np.float32).reshape(-1, 4))
        iou = box_iou(G, D) if len(G) and len(D) else torch.zeros(len(G), len(D))
        gc, dc = g.category_id.to_numpy(), p.category_id.to_numpy()
        matched, dtype = np.zeros(len(G), bool), []
        for k in range(len(D)):
            ov = (iou[:, k] >= iou_thr).numpy()
            same = ov & (gc == dc[k])
            if (same & ~matched).any():
                matched[np.argmax(iou[:, k].numpy() * (same & ~matched))] = True
                dtype.append("tp")
            elif same.any():
                dtype.append("duplicate")
            elif ov.any():
                dtype.append("wrong_class")
            else:
                cx, cy = float(D[k, 0] + D[k, 2]) / 2, float(D[k, 1] + D[k, 3]) / 2
                r = ig.get_group(i) if i in ig.groups else ignore.iloc[:0]
                inside = ((r.x <= cx) & (cx <= r.x + r.w) & (r.y <= cy) & (cy <= r.y + r.h)).any()
                dtype.append("ignored" if inside else "background")
        best = iou.max(1).values.numpy() if len(D) else np.zeros(len(G))
        arg = iou.argmax(1).numpy() if len(D) else np.zeros(len(G), int)
        pred_cls = np.array([CLASSES[dc[a] - 1] if len(D) else None for a in arg], dtype=object)
        overlap = np.where(pred_cls == g.cls.to_numpy(), "crowded", "wrong_class")
        out = np.where(matched, "correct", np.where(best >= iou_thr, overlap, np.where(best >= 0.1, "mislocalized", "missed")))
        gts.append(pd.DataFrame({"image_id": i, "cls": g.cls.to_numpy(), "size": g["size"].to_numpy(),
                                 "occlusion": g.occlusion.to_numpy(), "outcome": out, "pred_cls": pred_cls}))
        dets.append(pd.DataFrame({"image_id": i, "score": p.score.to_numpy(), "outcome": dtype}))
    gt = pd.concat(gts, ignore_index=True)
    gt["bucket"] = pd.cut(gt["size"], [0, 32, 96, 1e9], labels=["small", "medium", "large"])
    return gt, pd.concat(dets, ignore_index=True)


def error_tables(gt, det, show=True):
    """GT outcome by size and by occlusion (% of row) and detection types; displayed with captions if show."""
    by_size = pd.crosstab(gt.bucket, gt.outcome, normalize="index").mul(100).round(1).rename_axis("size (% of GT)")
    by_occ = pd.crosstab(gt.occlusion, gt.outcome, normalize="index").mul(100).round(1).rename_axis("occlusion (% of GT)")
    det_types = det.outcome.value_counts().rename("detections").to_frame().rename_axis("detection type")
    det_types["%"] = (det_types.detections / det_types.detections.sum() * 100).round(1)
    if show:
        from IPython.display import display
        cap = lambda df, t: df.style.set_caption(t).format("{:.1f}")
        display(cap(by_size, "GT boxes by size: outcome (% of row)"),
                cap(by_occ, "GT boxes by occlusion (0 none, 2 heavy): outcome (% of row)"),
                det_types.style.set_caption("Detections (score >= 0.3) by type").format({"detections": "{:d}", "%": "{:.1f}"}))
    return by_size, by_occ, det_types


def size_coverage(gt):
    """Share of GT boxes covered by a detection (IoU >= 0.5, any class) by COCO size bucket and overall,
    from the GT table of match_errors."""
    found = gt.outcome.isin(["correct", "crowded", "wrong_class"])
    r = found.groupby(gt.bucket.astype(str)).mean().reindex(["small", "medium", "large"])
    r["all"] = found.mean()
    return r


def plot_confusion(gt, title="Confusion matrix (GT boxes covered by a detection, % of row)"):
    """Confusion matrix over the GT boxes covered by a detection (IoU >= 0.5): correct class on the diagonal.
    Missed / mislocalized boxes have no matching detection and are not in it."""
    found = gt[gt.outcome.isin(["correct", "crowded", "wrong_class"])]
    pred = found.pred_cls.where(found.outcome == "wrong_class", found.cls)
    cm = pd.crosstab(found.cls, pred).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    cmp = cm.div(cm.sum(axis=1).replace(0, 1), axis=0).mul(100)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmp.values, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right"); ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm.values[i, j]:
                ax.text(j, i, f"{cmp.values[i, j]:.0f}%\n({cm.values[i, j]})", ha="center", va="center", fontsize=8,
                        color="white" if cmp.values[i, j] > 50 else "black")
    ax.set_xlabel("predicted class"); ax.set_ylabel("ground-truth class"); ax.set_title(title)
    plt.colorbar(im, fraction=0.046); plt.tight_layout(); plt.show()
    return cm


def plot_failure_frames(gt, det, preds, split="val", score_thr=0.3):
    """Frames with the most missed small objects, class confusions and background false positives:
    ground truth on top, predictions (coloured by predicted class) below. Returns the picked image ids."""
    per_img = gt.assign(missed_small=(gt.outcome == "missed") & (gt.bucket == "small"), confused=gt.outcome == "wrong_class") \
        .groupby("image_id")[["missed_small", "confused"]].sum()
    per_img["background_fp"] = det[det.outcome == "background"].groupby("image_id").size().reindex(per_img.index, fill_value=0)
    picks = {k: per_img[k].idxmax() for k in ["missed_small", "confused", "background_fp"]}
    handles = [plt.Line2D([], [], color=c, lw=3, label=n) for c, n in zip(COLORS, CLASSES)] + \
              [plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.35, label="ignore region")]
    for k, i in picks.items():
        fig, (gax, pax) = plt.subplots(2, 1, figsize=(16, 19))
        draw_frame(split, i, gax)
        draw_frame(split, i, pax, preds=preds, score_thr=score_thr, show_gt=False, color_preds=True)
        gax.set_title(f"{gax.get_title()} – most {k.replace('_', ' ')} ({per_img.loc[i, k]}) – ground truth", fontsize=12)
        pax.set_title(f"predictions (score ≥ {score_thr})", fontsize=12)
        fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=11)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.04, hspace=0.06); plt.show()
    return picks


__all__ = [
    "SEED",
    "ROOT",
    "COMPETITION",
    "HF_REPO",
    "ON_KAGGLE",
    "DATA_DIR",
    "RAW_DIR",
    "OUT_DIR",
    "RUNS_DIR",
    "SUBMISSIONS_DIR",
    "DEVICE",
    "CLASSES",
    "COLORS",
    "PREP",
    "EVAL_DETS",
    "seed_everything",
    "download_data",
    "load_split",
    "subsample",
    "coco_gt",
    "export_coco",
    "DetDataset",
    "collate",
    "make_loader",
    "tv_size_kwargs",
    "predict_dataset",
    "coco_evaluate",
    "evaluate_model",
    "torchvision_predictor",
    "fit",
    "plot_history",
    "RunLogger",
    "env_info",
    "run_dir",
    "save_run",
    "fetch_run",
    "load_results",
    "hf_token",
    "push_run",
    "pull_run",
    "push_file",
    "write_submission",
    "check_submission",
    "make_submission",
    "kaggle_secrets",
    "submit_to_kaggle",
    "check_hf_write",
    "draw_frame",
    "anchor_fit",
    "anchor_fit_by_size",
    "match_errors",
    "error_tables",
    "size_coverage",
    "plot_confusion",
    "plot_failure_frames",
]
