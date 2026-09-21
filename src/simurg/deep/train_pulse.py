# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# train_pulse — trains SIMURG Pulse (the optional deep-learning tier).
# Data sources:
#   1. live endpoint: SIMURG_LIVE_URL + SIMURG_LIVE_MODEL -> real answers from
#      the model being guarded (clean, label 0),
#   2. CorruptBench: synthetic corruptions of the clean corpus (label 1, exact
#      onset known).
# Windows are sampled every STEP chars along each stream (as a live checkpoint
# sees them). Labels are ONSET-AWARE: a window is corrupt only when its right
# edge lies past onset + LABEL_MARGIN (mirrors data/evaluate._train_model).
# Objective: BCE-with-logits weighted by pos_weight (robust, well-tested).
# After training, calibration anchors (lo/hi) come from held-out raw scores and
# are stored in the safetensors metadata. A standalone AUROC (Mann-Whitney rank
# statistic) reports generalization.
#
# Run:
#   SIMURG_LIVE_URL=http://host:port/v1/chat/completions #   SIMURG_LIVE_MODEL=wahoo-1.5-preview #   python3 -m simurg.deep.train_pulse --clean 40 --corrupt 240 --epochs 8
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.request

import numpy as np

from ..core import DRIFT, REGURGITATION, REPETITION, STRUCTURAL
from ..data.synth import corrupt
from ..data.dataset import load_clean_corpus
from .pulse import build_net, tokenize, weights_path

STEP = 150
MIN_CHARS = 400
LABEL_MARGIN = 300


def sample_windows(text: str) -> list:
    out = []
    for pos in range(STEP, len(text) - MIN_CHARS + 1, STEP):
        out.append((tokenize(text[:pos]), pos))
    return out


def generate_live(url: str, model: str, prompts, max_tokens: int = 512) -> list:
    texts = []
    for i, p in enumerate(prompts):
        body = {"model": model, "stream": False, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": p}]}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                doc = json.loads(resp.read().decode())
                txt = doc["choices"][0]["message"]["content"]
            if len(txt) >= 300:
                texts.append(txt)
        except Exception as e:
            print(f"  live sample {i} failed: {str(e)[:120]}")
        time.sleep(0.2)
    return texts


def _default_prompts(n: int) -> list:
    topics = ["Explain how oil prices affect a small open economy.",
              "Summarize the quarterly macro report for a non-technical reader.",
              "Describe the history of ancient Ur in a few paragraphs.",
              "Write a product launch announcement for a smart thermostat.",
              "Explain inflation targeting to a first-year student.",
              "Describe how a small bakery can grow via social media.",
              "Summarize the impact of fiscal stimulus on construction.",
              "Explain the current account in simple terms.",
              "Write an overview of renewable energy adoption.",
              "Describe the risks of a banking-sector credit squeeze."]
    return [topics[i % len(topics)] for i in range(n)]


def _auroc(scores, labels) -> float:
    """Mann-Whitney rank statistic; returns 0.5 for no separation."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    pos = ranks[labels == 1]
    n_pos, n_neg = len(pos), len(labels) - len(pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((pos.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def train(args) -> str:
    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    device = torch.device(args.device or "cpu")
    print(f"train device: {device}")

    rng = random.Random(args.seed)
    clean_pool = [t for t in load_clean_corpus() if len(t) >= 700]
    live_texts = []
    if args.live_url and args.live_model:
        prompts = None
        if args.prompts_jsonl and os.path.exists(args.prompts_jsonl):
            with open(args.prompts_jsonl, encoding="utf-8") as f:
                prompts = [json.loads(l)["text"] for l in f if l.strip()][:args.clean]
        if not prompts:
            prompts = _default_prompts(args.clean)
        print(f"generating {len(prompts)} live samples from {args.live_model} ...")
        live_texts = generate_live(args.live_url, args.live_model, prompts)
        print(f"  got {len(live_texts)} usable live texts")
    all_clean = live_texts + clean_pool
    rng.shuffle(all_clean)
    n_train_clean = max(1, int(len(all_clean) * 0.8))
    train_clean, val_clean = all_clean[:n_train_clean], all_clean[n_train_clean:] or all_clean[:1]

    # ── train tensors (onset-aware) ─────────────────────────────────────────
    X, y = [], []
    for t in train_clean:
        for w, _pos in sample_windows(t):
            X.append(w); y.append(0.0)
    n_corr_per = max(1, args.corrupt // 4)
    for cls in (REPETITION, DRIFT, REGURGITATION, STRUCTURAL):
        for _ in range(n_corr_per):
            base = rng.choice(train_clean)
            text, _onset, _c = corrupt(base, cls, rng)
            onset = int(_onset or 0)
            for w, pos in sample_windows(text):
                if pos >= onset + LABEL_MARGIN:
                    X.append(w); y.append(1.0)
                elif pos <= onset:
                    X.append(w); y.append(0.0)
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)
    n_pos = int(y.sum())
    pos_weight = torch.tensor(float(len(y) - n_pos) / max(1, n_pos))
    print(f"train windows: {len(y)} ({n_pos} corrupt, pos_weight {float(pos_weight):.2f})")

    # ── validation tensors (onset-aware) ────────────────────────────────────
    vX, vy = [], []
    for t in val_clean:
        for w, _pos in sample_windows(t):
            vX.append(w); vy.append(0.0)
    for cls in (REPETITION, DRIFT):
        for _ in range(20):
            base = rng.choice(val_clean)
            text, _o, _c = corrupt(base, cls, rng)
            onset = int(_o or 0)
            for w, pos in sample_windows(text):
                if pos >= onset + LABEL_MARGIN:
                    vX.append(w); vy.append(1.0)
                elif pos <= onset:
                    vX.append(w); vy.append(0.0)
    vX = torch.tensor(vX, dtype=torch.long).to(device)
    vy = torch.tensor(vy, dtype=torch.float32).to(device)

    # ── model / training ────────────────────────────────────────────────────
    net = build_net(torch).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params: {n_params:,}")
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(weight=pos_weight.to(device))
    bs = args.batch
    n = len(y)
    best_val = 1e9
    best_sd = None
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        n_batches = 0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = X[idx].to(device); yb = y[idx].to(device)
            logits = net(xb)
            loss = bce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n_batches += 1
        with torch.no_grad():
            vp = torch.sigmoid(net(vX))
            vloss = bce(torch.logit(vp.clamp(1e-6, 1 - 1e-6)), vy)
            vpred = vp.mean().item(); vmin = vp.min().item(); vmax = vp.max().item()
        print(f"epoch {epoch}: bce {tot / max(1, n_batches):.4f} | val bce {float(vloss):.4f} "
              f"| val pred mean {vpred:.3f} [{vmin:.3f},{vmax:.3f}]")
        if float(vloss) < best_val:
            best_val = float(vloss)
            best_sd = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    if best_sd is not None:
        net.load_state_dict(best_sd)

    # ── calibration anchors + AUROC from held-out raw scores ────────────────
    with torch.no_grad():
        vraw = torch.sigmoid(net(vX)).cpu().numpy()
    vlab = vy.cpu().numpy()
    lo = float(np.percentile(vraw[vlab == 0], 95))
    hi = float(np.median(vraw[vlab == 1]))
    if not (lo < hi):
        lo, hi = float(np.min(vraw)), float(np.max(vraw))
    auroc = _auroc(vraw, vlab)
    print(f"held-out AUROC: {auroc:.3f} | anchors lo={lo:.3f} hi={hi:.3f}")

    from .pulse import PulseModel
    pm = PulseModel(torch, net, device, lo, hi)
    out = args.out or weights_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pm.save(out, meta={"auroc_heldout": round(auroc, 4),
                       "n_train_windows": int(len(y)), "n_live": len(live_texts)})
    print(f"saved -> {out} ({os.path.getsize(out) / 1024 / 1024:.2f} MB)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=int, default=40)
    ap.add_argument("--corrupt", type=int, default=240)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--live-url", default=os.environ.get("SIMURG_LIVE_URL", ""))
    ap.add_argument("--live-model", default=os.environ.get("SIMURG_LIVE_MODEL", ""))
    ap.add_argument("--prompts-jsonl", default=os.environ.get("SIMURG_LIVE_PROMPTS_JSONL", ""))
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="",
                    help="training device: cpu (default) or mps/cuda")
    train(ap.parse_args())
