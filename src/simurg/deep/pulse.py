# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# pulse — SIMURG Pulse, the optional deep-learning tier. A small streaming
# transformer (~350K params, safetensors) that reads the recent character
# window and outputs a calibrated corruption probability. It captures
# SEQUENTIAL structure (loop phase, script-switch cadence, garbage texture)
# that the 15-dim statistics of the numpy tier compress away. Runs on CPU or
# Apple Silicon (MPS); one 256-token forward is single-digit milliseconds, so
# a checkpoint every 400 chars stays far below the model's token rate.
#
# Optional by design: the numpy-only core never imports torch. When torch +
# safetensors + a trained weights file are present, the pulse detector joins
# the ensemble automatically; otherwise it is silently absent and every
# existing behaviour is unchanged.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import math
import os

# ── fixed architecture constants (part of the weights-file contract) ────────
VOCAB = 4096        # trigram hash buckets (+1 pad slot at index 0)
SEQ = 256           # max trigrams per window
DIM = 64            # embedding / hidden width
DEPTH = 2           # transformer encoder layers
HEADS = 4
TAIL_CHARS = 600    # mirror StreamFeatures.tail maxlen
WEIGHTS_NAME = "simurg_pulse.safetensors"


def weights_path() -> str:
    """Where the trained Pulse weights live (env override for tests/customs)."""
    env = os.environ.get("SIMURG_PULSE_WEIGHTS")
    if env:
        return env
    simurg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(simurg_dir, "weights", WEIGHTS_NAME)


def _trigram_token(t: str) -> int:
    from ..core import stable_hash
    # map into 1..VOCAB-1 so index 0 stays reserved for left padding
    return (stable_hash(t) % (VOCAB - 1)) + 1


def tokenize(text: str, seq: int = SEQ) -> list:
    """The last seq character-trigram tokens of text, left-padded with 0s.

    Deterministic across processes (blake2b via core.stable_hash), so a
    training window and a live checkpoint over the same text produce the same
    tensor. O(len(text))."""
    if len(text) < 3:
        return [0] * seq
    toks = [_trigram_token(text[i:i + 3]) for i in range(len(text) - 2)]
    toks = toks[-seq:]
    return [0] * (seq - len(toks)) + toks


def build_net(torch):
    """The Pulse architecture as a fresh (untrained) nn.Module. Constants are
    fixed at module level so any safetensors file matches this shape."""

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(VOCAB, DIM, padding_idx=0)
            self.pos = torch.nn.Parameter(torch.zeros(1, SEQ, DIM))
            layer = torch.nn.TransformerEncoderLayer(
                DIM, HEADS, DIM * 2, 0.05, batch_first=True, norm_first=True)
            self.enc = torch.nn.TransformerEncoder(layer, DEPTH)
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(DIM), torch.nn.Linear(DIM, 1))

        def forward(self, x):
            h = self.emb(x) + self.pos[:, :x.size(1)]
            mask = (x == 0)
            h = self.enc(h, src_key_padding_mask=mask)
            keep = (~mask).float().unsqueeze(-1)
            h = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
            return self.head(h).squeeze(-1)

    return Net()


def _pick_device(torch):
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PulseModel:
    """Torch-free wrapper: loads a safetensors file, runs calibration-mapped
    inference. prob(tokens) -> float in [0, 1]."""

    def __init__(self, torch, net, device, lo: float, hi: float):
        self.torch = torch
        self.net = net
        self.device = device
        self.lo = lo
        self.hi = hi
        self.net.eval()

    def prob(self, tokens) -> float:
        torch = self.torch
        x = torch.tensor([tokens], dtype=torch.long, device=self.device)
        with torch.no_grad():
            z = float(self.net(x).item())
        s = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        # linear calibration: lo = high quantile of CLEAN scores, hi = median
        # of CORRUPT scores, so the mapped probability is comparable with the
        # rule/ngram/sketch tiers that the conformal thresholds assume.
        span = max(1e-6, self.hi - self.lo)
        return max(0.0, min(1.0, (s - self.lo) / span))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    # ── persistence (safetensors) ───────────────────────────────────────────
    def save(self, path: str, meta: dict | None = None) -> None:
        from safetensors.torch import save_file
        sd = {k: v.detach().contiguous() for k, v in self.net.state_dict().items()}
        blob = {"lo": self.lo, "hi": self.hi, **(meta or {})}
        save_file(sd, path, metadata={"simurg_pulse": json.dumps(blob)})

    @classmethod
    def load(cls, path: str, torch=None) -> "PulseModel":
        from safetensors.torch import load_file
        if torch is None:
            import torch
        state = load_file(path, device="cpu")
        meta = _read_safetensors_meta(path)
        net = build_net(torch)
        net.load_state_dict(state)
        info = json.loads(meta.get("simurg_pulse", "{}")) if meta else {}
        device = _pick_device(torch)
        net.to(device)
        return cls(torch, net, device,
                   float(info.get("lo", 0.3)), float(info.get("hi", 0.8)))


def _read_safetensors_meta(path: str) -> dict:
    """Read the JSON metadata block from a safetensors header without loading
    tensor payloads. Format: <u64 header_len><json header> ... The metadata
    dict lives under the special "__metadata__" key of the header."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return header.get("__metadata__") or {}
