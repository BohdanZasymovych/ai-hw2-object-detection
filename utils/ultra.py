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
import random
import shutil
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision.ops import clip_boxes_to_image

from utils.common import (CLASSES, DATA_DIR, DEVICE, EVAL_DETS, HF_REPO, ON_KAGGLE, OUT_DIR, PREP, RAW_DIR, ROOT, SEED,
                          hf_token, load_split, run_dir, subsample)

ULTRA_VERSION = "8.4.207"                                     # tested version; the patches depend on its internals
ULTRA_DIR = OUT_DIR / "ultralytics"                           # Ultralytics run folders and downloaded weights
YOLO_DATA_DIR = Path("/tmp/yolo_data") if ON_KAGGLE else DATA_DIR / "yolo"   # Kaggle: local disk, not /kaggle/working
MODEL_CFG_DIR = ROOT / "utils" / "models"


# --------------------------------------------------------------------------------------
# Dataset export
# --------------------------------------------------------------------------------------
def _read_yolo(path, cols):
    """Rows of a YOLO txt file as an array with `cols` columns (empty if the file is missing or empty)."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, cols))
    return np.loadtxt(path, ndmin=2).reshape(-1, cols)


def _covered(boxes, regions):
    """Largest share of each normalised cxcywh box covered by one of the normalised cxcywh regions."""
    if not len(boxes) or not len(regions):
        return np.zeros(len(boxes))
    b, r = boxes[:, None], regions[None]
    ix = np.minimum(b[..., 0] + b[..., 2] / 2, r[..., 0] + r[..., 2] / 2) - np.maximum(b[..., 0] - b[..., 2] / 2, r[..., 0] - r[..., 2] / 2)
    iy = np.minimum(b[..., 1] + b[..., 3] / 2, r[..., 1] + r[..., 3] / 2) - np.maximum(b[..., 1] - b[..., 3] / 2, r[..., 1] - r[..., 3] / 2)
    return (ix.clip(min=0) * iy.clip(min=0)).max(1) / (boxes[:, 2] * boxes[:, 3])


def _crop_labels(labels, size, window, min_visible=0.3, min_px=2):
    """Normalised YOLO labels of a frame of `size` (w, h) -> labels of the crop `window` (x1, y1, x2, y2). Boxes with
    less than `min_visible` of their area inside the crop or a side below `min_px` are dropped."""
    w, h = size
    x1, y1, x2, y2 = window
    cx, cy, bw, bh = labels[:, 1] * w, labels[:, 2] * h, labels[:, 3] * w, labels[:, 4] * h
    a0, b0, a1, b1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
    a, b, c, d = a0.clip(x1, x2), b0.clip(y1, y2), a1.clip(x1, x2), b1.clip(y1, y2)
    keep = ((c - a) * (d - b) >= min_visible * bw * bh) & (c - a >= min_px) & (d - b >= min_px)
    tw, th = x2 - x1, y2 - y1
    return np.c_[labels[:, 0], ((a + c) / 2 - x1) / tw, ((b + d) / 2 - y1) / th, (c - a) / tw, (d - b) / th][keep]


def _resize_one(task):
    src, dst, lbl_src, lbl_dst, long_side, opts = task
    mask_ignore, degrade_to, tile = opts.get("mask_ignore"), opts.get("degrade_to"), opts.get("tile")
    im = Image.open(src)
    ow, oh = im.size
    s = long_side / max(ow, oh)
    size = (round(ow * s), round(oh * s))
    regions = _read_yolo(lbl_src.parents[2] / "ignore" / lbl_src.parent.name / lbl_src.name, 4) if mask_ignore else None
    window = None
    if tile:
        from ultralytics.data.utils import tile_windows
        windows = tile_windows(size[::-1], tile)
        window = windows[opts["index"] % len(windows)]
    if not dst.exists():
        im.draft("RGB", size)                                    # fast JPEG decoding at reduced size
        im = im.convert("RGB")
        if degrade_to:                                           # low-resolution frame: lost detail stays lost when enlarged
            k = degrade_to / max(ow, oh)
            im = im.resize((round(ow * k), round(oh * k)), Image.BILINEAR)
        im = im.resize(size, Image.BILINEAR)
        if mask_ignore:
            draw = ImageDraw.Draw(im)
            for cx, cy, w, h in regions:
                draw.rectangle([(cx - w / 2) * size[0], (cy - h / 2) * size[1], (cx + w / 2) * size[0], (cy + h / 2) * size[1]],
                               fill=(114, 114, 114))
        if window:
            im = im.crop(window)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=95)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    if mask_ignore or tile:
        labels = _read_yolo(lbl_src, 5)
        if mask_ignore:                                          # a box mostly under the grey paint is no longer visible
            labels = labels[_covered(labels[:, 1:], regions) <= 0.5]
        if tile:
            labels = _crop_labels(labels, size, window)
        lbl_dst.write_text("".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels))
    elif lbl_src.exists():                                       # YOLO labels are normalised: resizing keeps them valid
        shutil.copyfile(lbl_src, lbl_dst)
    else:
        lbl_dst.write_text("")


def _lowres_target(file_name, lowres):
    """Long side a train frame is degraded to ({long side: share of frames}), or None. Fixed per frame by SEED."""
    r = random.Random(f"{SEED}:{file_name}").random()
    for side, share in (lowres or {}).items():
        if r < share:
            return side
        r -= share
    return None


def export_yolo(out_dir=YOLO_DATA_DIR, step=PREP["frame_step"], long_side=PREP["long_side"], max_frames=None,
                workers=4, trainval=False, mask_ignore=False, lowres=None, tile=None):
    """Ultralytics dataset of the subsampled train/val frames, resized once to `long_side` (same frames and
    resolution as DetDataset). Decoding the native frames (up to 3840x2160) in every epoch would make the 4 Kaggle
    CPUs the bottleneck. Labels: the provided YOLO txt files (8 classes, iscrowd=0 only; ignore regions are not
    targets). max_frames: the same reproducible subset as DetDataset (quick tests). trainval=True: the training list
    holds train + val (final model only: its validation metrics are then meaningless). Returns the data yaml path.
    Data strategies: mask_ignore paints ignore regions grey (train only); lowres ({long side: share}) enlarges that
    share of train frames from a smaller size (train only); tile (px) keeps one crop per frame, cycling through the
    overlapping windows of `tile_windows` along each sequence, so an epoch has as many images as without tiling."""
    out_dir = Path(out_dir)
    t0, tasks, lists = time.time(), [], {}
    for split in ["train", "val"]:
        frames = subsample(load_split(split)[0], step)
        if max_frames:
            frames = frames.sample(min(max_frames, len(frames)), random_state=SEED).sort_index()
        paths = []
        positions = frames.groupby("seq").cumcount()
        for f, index in zip(frames.file_name, positions):        # images/<seq>/<frame>.jpg
            dst = out_dir / split / f
            lbl = Path(f.replace("images/", "labels/", 1)).with_suffix(".txt")
            opts = dict(tile=tile, index=int(index))
            if split == "train":
                opts.update(mask_ignore=mask_ignore, degrade_to=_lowres_target(f, lowres))
            tasks.append((RAW_DIR / split / f, dst, RAW_DIR / split / lbl, out_dir / split / lbl, long_side, opts))
            paths.append(str(dst.resolve()))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{split}.txt").write_text("\n".join(paths))
        lists[split] = paths
    with get_context("fork").Pool(workers) as pool:
        for _ in pool.imap_unordered(_resize_one, tasks, chunksize=32):
            pass
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    train_list, yaml = "train.txt", out_dir / "data.yaml"
    if trainval:
        train_list, yaml = "trainval.txt", out_dir / "data_trainval.yaml"
        (out_dir / train_list).write_text("\n".join(lists["train"] + lists["val"]))
    yaml.write_text(f"path: {out_dir.resolve()}\ntrain: {train_list}\nval: val.txt\nnames:\n{names}")
    print(f"YOLO dataset: {len(tasks)} frames at {long_side} px in {out_dir} ({time.time() - t0:.0f} s)")
    return yaml


# --------------------------------------------------------------------------------------
# Models and experiment patches
# --------------------------------------------------------------------------------------
def model_yaml(arch="yolo11s", p2=False):
    """Model config: the Ultralytics yaml of `arch`, or our P2 variant (utils/models/yolo11-p2.yaml; the scale letter
    in the file name selects the model size, as in Ultralytics)."""
    return str(MODEL_CFG_DIR / f"{arch}-p2.yaml") if p2 else f"{arch}.yaml"


def is_rtdetr(arch):
    return arch.startswith("rtdetr")


def rtdetr_cfg(arch="rtdetr-l", nd=100):
    """RT-DETR model config (dict) with `nd` denoising queries; the other decoder arguments keep the Ultralytics
    defaults. Changing nd adds no weights, so the COCO checkpoint still fits (only the class layers are new)."""
    import ultralytics
    from ultralytics.utils import YAML
    cfg = YAML.load(Path(ultralytics.__file__).parent / "cfg" / "models" / "rt-detr" / f"{arch}.yaml")
    # RTDETRDecoder(nc, hd, nq, ndp, nh, ndl, d_ffn, dropout, act, eval_idx, nd); the channels are inserted by Ultralytics
    cfg["head"][-1][-1] = ["nc", 256, 300, 4, 8, 6, 1024, 0.0, None, -1, nd]
    return cfg


def nwd(pred, target, c):
    """Normalized Wasserstein Distance similarity (Wang et al., 2021) of xyxy boxes in pixels: each box is a 2D
    Gaussian, W2^2 = |centre difference|^2 + |(w, h) difference|^2 / 4, similarity exp(-W2 / c). Unlike IoU it
    changes smoothly with a few-pixel shift of a tiny box and is non-zero for non-overlapping boxes."""
    cp, ct = (pred[:, :2] + pred[:, 2:]) / 2, (target[:, :2] + target[:, 2:]) / 2
    wp, wt = pred[:, 2:] - pred[:, :2], target[:, 2:] - target[:, :2]
    w2 = (cp - ct).pow(2).sum(-1) + ((wp - wt) / 2).pow(2).sum(-1)
    return torch.exp(-w2.clamp(min=1e-7).sqrt() / c)


def apply_patches(nwd_ratio=0.0, nwd_c=32.0, backbone_lr_ratio=1.0):
    """Experiment changes, applied in the training process before the trainer builds the loss and the optimizer.
    nwd_ratio: YOLO box loss = (1 - r) * CIoU loss + r * NWD loss (r = 0: Ultralytics default).
    nwd_c: NWD normalising constant in input pixels (about the median box size, notebook 01).
    backbone_lr_ratio: learning rate of the backbone layers relative to the rest of the model (1: single rate)."""
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.torch_utils import unwrap_model

    if backbone_lr_ratio != 1.0:
        orig_build = BaseTrainer.build_optimizer

        def build_optimizer(self, model, *args, **kwargs):
            # Split every parameter group of Ultralytics (bias, weight, bn) into its non-backbone and backbone parts
            optimizer = orig_build(self, model, *args, **kwargs)
            net = unwrap_model(model)
            backbone = {id(p) for layer in net.model[:len(net.yaml["backbone"])] for p in layer.parameters()}
            groups = list(optimizer.param_groups)
            optimizer.param_groups.clear()
            for g in groups:
                rest = [p for p in g["params"] if id(p) not in backbone]
                back = [p for p in g["params"] if id(p) in backbone]
                for params, lr in [(rest, g["lr"]), (back, g["lr"] * backbone_lr_ratio)]:
                    if params:
                        optimizer.add_param_group({**g, "params": params, "lr": lr})
            return optimizer

        BaseTrainer.build_optimizer = build_optimizer

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
    """Body of the training subprocess. cfg: name, data, arch, weights, train (Ultralytics train arguments), use_last
    (keep the last epoch instead of the best one, for runs without a clean val set); YOLO: p2, nwd_ratio, nwd_c;
    RT-DETR: nd, backbone_lr_ratio. Writes best.pt, results.csv, args.yaml and history.json into runs/<name>/."""
    from ultralytics import RTDETR, YOLO
    from ultralytics.utils import YAML
    apply_patches(cfg.get("nwd_ratio", 0.0), cfg.get("nwd_c", 32.0), cfg.get("backbone_lr_ratio", 1.0))
    rtdetr = is_rtdetr(cfg["arch"])
    model_cls = RTDETR if rtdetr else YOLO
    save = ULTRA_DIR / cfg["name"]
    last = save / "weights" / "last.pt"
    if last.exists():                                            # interrupted run in this session: continue it
        try:
            model_cls(str(last)).train(resume=True)
        except AssertionError as e:                              # already finished, only the copy below is missing
            print(e)
    else:
        if rtdetr:                                               # one file per run: two runs may start at the same time
            yaml = run_dir(cfg["name"]) / "model.yaml"
            YAML.save(yaml, rtdetr_cfg(cfg["arch"], cfg["nd"]))
        else:
            yaml = model_yaml(cfg["arch"], cfg["p2"])
        model = model_cls(str(yaml)).load(cfg["weights"])
        model.train(data=cfg["data"], project=str(ULTRA_DIR), name=cfg["name"], exist_ok=True,
                    device=0 if torch.cuda.is_available() else "cpu", **cfg["train"])
    d = run_dir(cfg["name"])
    for f in ["results.csv", "args.yaml"]:
        shutil.copyfile(save / f, d / f)
    json.dump(_history(save / "results.csv"), open(d / "history.json", "w"), indent=1)
    weights = "last.pt" if cfg.get("use_last") else "best.pt"
    shutil.copyfile(save / "weights" / weights, d / "best.pt")   # written last: marks the run as finished


CHECKPOINT_FILES = ("results.csv", "weights/best.pt", "weights/last.pt")     # last.pt last: it marks a resumable run


def _push_checkpoint(name, pushed):
    """Upload the training state of a running run to the HF repo (runs/<name>/ckpt/), so a lost session loses little.
    pushed: {file: modification time already uploaded}. A failed upload is only reported: training must go on."""
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())
    for rel in CHECKPOINT_FILES:
        f = ULTRA_DIR / name / rel
        if not f.exists() or pushed.get(rel) == f.stat().st_mtime or time.time() - f.stat().st_mtime < 30:
            continue                                             # missing, unchanged or possibly still being written
        mtime, tmp = f.stat().st_mtime, f.with_name(f.name + ".upload")
        try:
            shutil.copyfile(f, tmp)
            api.upload_file(path_or_fileobj=str(tmp), path_in_repo=f"runs/{name}/ckpt/{Path(rel).name}", repo_id=HF_REPO,
                            commit_message=f"checkpoint {name}")
            pushed[rel] = mtime
        except Exception as e:
            print(f"{name}: checkpoint upload of {rel} failed ({type(e).__name__}: {e}); retrying later")
        finally:
            tmp.unlink(missing_ok=True)


def _pull_checkpoint(name):
    """Restore the training state uploaded by an earlier session of this run, if any, so that _train resumes it."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    if (ULTRA_DIR / name / "weights" / "last.pt").exists():
        return
    try:
        for rel in CHECKPOINT_FILES:
            src = hf_hub_download(HF_REPO, f"runs/{name}/ckpt/{Path(rel).name}", token=hf_token())
            dst = ULTRA_DIR / name / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        print(f"{name}: resuming from the checkpoint in the HF repo")
    except EntryNotFoundError:                                   # nothing uploaded yet: a fresh start
        pass


def train_runs(cfgs, checkpoint_min=None):
    """Train the given runs (list of cfg dicts for _train), skipping finished ones (runs/<name>/best.pt exists).
    With 2+ GPUs the runs are trained concurrently, one per GPU; otherwise one after another. Output of each run
    goes to runs/<name>/train.log; a short summary is printed at the end.
    checkpoint_min: upload the training state to the HF repo every this many minutes and resume from it if a run was
    interrupted in an earlier session (for long runs)."""
    todo = [c for c in cfgs if not (run_dir(c["name"]) / "best.pt").exists()]
    for c in cfgs:
        if c not in todo:
            print(f"{c['name']}: already trained, skipped")
    n_gpus = max(torch.cuda.device_count(), 1)
    ULTRA_DIR.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset
    for w in {c["weights"] for c in todo}:                       # once, before concurrent runs could both start the download
        attempt_download_asset(str(ULTRA_DIR / w))
    env = {**os.environ, "PYTHONPATH": str(ROOT), "WANDB_MODE": "disabled"}
    for i in range(0, len(todo), n_gpus):
        procs, t0 = [], time.time()
        for gpu, c in enumerate(todo[i:i + n_gpus]):
            if checkpoint_min:
                _pull_checkpoint(c["name"])
            log = open(run_dir(c["name"]) / "train.log", "a" if checkpoint_min else "w")
            gpu_env = {"CUDA_VISIBLE_DEVICES": str(gpu)} if n_gpus > 1 else {}   # pin a GPU only when sharing them
            p = subprocess.Popen([sys.executable, "-m", "utils.ultra", json.dumps(c, default=str)], cwd=ULTRA_DIR,
                                 env={**env, **gpu_env}, stdout=log, stderr=subprocess.STDOUT)
            procs.append((c["name"], gpu, p, log))
            print(f"{c['name']}: training on GPU {gpu}, log in {run_dir(c['name']) / 'train.log'}")
        pushed, last_push = {name: {} for name, *_ in procs}, time.time()
        while checkpoint_min and any(p.poll() is None for _, _, p, _ in procs):
            time.sleep(30)
            if time.time() - last_push >= checkpoint_min * 60:
                for name, _, p, _ in procs:
                    if p.poll() is None:
                        _push_checkpoint(name, pushed[name])
                last_push = time.time()
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
# Speed test (RT-DETR)
# --------------------------------------------------------------------------------------
def rtdetr_speed(imgsz, batch=4, nd=100, n_gt=46, iters=30, arch="rtdetr-l"):
    """Training cost of RT-DETR at a square input on synthetic data (forward, loss with matching, backward, optimizer
    step; no data loading), run before the real training so that it can set the input size. n_gt: boxes per image.
    Returns imgsz, ms per image (median, without the first 2 iterations) and peak GPU memory; ms is inf if out of memory."""
    from ultralytics.nn.tasks import RTDETRDetectionModel
    use_gpu = DEVICE.type == "cuda"
    net = RTDETRDetectionModel(rtdetr_cfg(arch, nd), nc=len(CLASSES), verbose=False).to(DEVICE).train()
    net.nc = len(CLASSES)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=use_gpu)
    g = torch.Generator().manual_seed(SEED)
    n = batch * n_gt
    boxes = torch.cat([torch.rand(n, 2, generator=g) * 0.8 + 0.1, torch.rand(n, 2, generator=g) * 0.04 + 0.01], 1)
    data = dict(img=torch.rand(batch, 3, imgsz, imgsz, generator=g).to(DEVICE), bboxes=boxes.to(DEVICE),
                cls=torch.randint(0, len(CLASSES), (n, 1), generator=g).float().to(DEVICE),
                batch_idx=torch.arange(batch).repeat_interleave(n_gt).float().to(DEVICE))
    times = []
    try:
        if use_gpu:
            torch.cuda.reset_peak_memory_stats()
        for _ in range(iters):
            t0 = time.perf_counter()
            with torch.autocast(DEVICE.type, enabled=use_gpu):
                loss, _ = net.loss(data)
            scaler.scale(loss.sum()).backward()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            if use_gpu:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        ms = 1000 * float(np.median(times[2:] or times)) / batch
    except torch.cuda.OutOfMemoryError:
        ms = float("inf")
    peak = torch.cuda.max_memory_allocated() / 1e9 if use_gpu else 0.0
    del net, opt, data
    torch.cuda.empty_cache()
    return dict(imgsz=imgsz, ms_per_img=ms, peak_gb=peak)


# --------------------------------------------------------------------------------------
# Inference for the shared COCO evaluation
# --------------------------------------------------------------------------------------
def load_ultra(path):
    """Trained Ultralytics model (YOLO or RT-DETR) from a best.pt."""
    from ultralytics import RTDETR, YOLO
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    is_detr = "RTDETR" in type(ckpt["model"]).__name__
    return (RTDETR if is_detr else YOLO)(str(path))


def ultra_predictor(model, imgsz=PREP["long_side"], nms=0.7, half=True, tile=False):
    """predict_fn for common.predict_dataset: DetDataset images (float RGB in [0, 1], already resized to imgsz on the
    long side) -> boxes in the same pixels, scores and 0-based labels (use label_offset=0). Ultralytics only pads the
    short side to a multiple of the stride. nms: IoU of the final NMS (Ultralytics default 0.7).
    RT-DETR: Ultralytics stretches each frame to imgsz x imgsz and there is no NMS (nms is ignored), so the frames must
    keep at least imgsz on the long side; if they were resized to it first, the height would be upsampled from less.
    Boxes are clipped to the frame: Ultralytics does this for YOLO, but not for RT-DETR.
    tile (px): predict on overlapping crops of this size; Ultralytics merges them with NMS at IoU `nms`."""
    dev = 0 if DEVICE.type == "cuda" else "cpu"

    def predict(images):
        arrs = [np.ascontiguousarray((im.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)[:, :, ::-1])
                for im in images]                                # Ultralytics expects BGR uint8 (OpenCV convention)
        res = model.predict(arrs, imgsz=imgsz, conf=0.001, iou=nms, max_det=EVAL_DETS, tile=tile,
                            quantize=16 if half and dev == 0 else None,   # 16: FP16 inference ('half' is deprecated)
                            device=dev, verbose=False)
        return [dict(boxes=clip_boxes_to_image(r.boxes.xyxy, r.orig_shape), scores=r.boxes.conf, labels=r.boxes.cls.long())
                for r in res]
    return predict


if __name__ == "__main__":                                       # python -m utils.ultra '<cfg json>'
    _train(json.loads(sys.argv[1]))
