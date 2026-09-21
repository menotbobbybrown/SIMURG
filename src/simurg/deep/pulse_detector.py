# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# pulse_detector — the ensemble member that wraps SIMURG Pulse. Registered
# unconditionally; at evaluate-time it lazily loads the safetensors weights
# (and torch/MPS). If any of torch, safetensors, or a trained weights file is
# missing, it degrades to a silent p=0 contribution, so the numpy-only core
# behaves EXACTLY as before. When all three are present, pulse joins the
# ensemble automatically and its score participates in the same conformal
# fusion as the other five detectors.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

from ..core import REPETITION, DRIFT, REGURGITATION, STRUCTURAL, REGISTRY, DetectorScore

_MIN_CHARS = 400      # need enough context before a window is meaningful


@REGISTRY.register("pulse")
def _pulse_factory():
    return PulseDetector()


class PulseDetector:
    """Deep tier: corruption probability from a small transformer over the
    recent character window. Lazy + fault-tolerant by contract."""
    name = "pulse"

    def __init__(self):
        self._model = None
        self._failed = False

    def _ensure(self):
        if self._model is not None or self._failed:
            return
        try:
            import torch
            import os
            from .pulse import PulseModel, weights_path
            path = weights_path()
            if not os.path.exists(path):
                self._failed = True
                return
            self._model = PulseModel.load(path, torch)
        except Exception:
            self._failed = True

    def evaluate(self, state) -> DetectorScore:
        self._ensure()
        if self._model is None:
            return DetectorScore(self.name, 0.0)
        if state.total_len < _MIN_CHARS:
            return DetectorScore(self.name, 0.0)
        try:
            from .pulse import tokenize
            window = "".join(state.tail)
            p = self._model.prob(tokenize(window))
        except Exception:
            return DetectorScore(self.name, 0.0)
        # interpret as a corruption probability with class hints: the model was
        # trained on all four corrupt classes, so attribute to the dominant
        # stream signal when it fires strongly.
        reasons = []
        classes = []
        if p >= 0.7:
            reasons.append(f"pulse deep-tier p={p:.2f}")
            s = state.snapshot()
            if s.get("repeat_rate", 0) > 0.4:
                classes.append(REPETITION)
            elif s["foreign_frac"] > 0.05:
                classes.append(DRIFT)
            elif s["structural_density"] > 0.6 or s["digit_frac"] > 0.2:
                classes.append(STRUCTURAL)
            else:
                classes.append(REGURGITATION)
        return DetectorScore(self.name, p, reasons, classes)
