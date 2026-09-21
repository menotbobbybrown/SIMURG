# SIMURG Pulse — Streaming Corruption Detector (deep tier)

A small 2-layer transformer that streams over the last ~600 characters of an
LLM response and outputs a calibrated corruption probability. It runs in a
few milliseconds on Apple Silicon and adds recall headroom for sequential
pathologies (loop phase, script cadence, garbage texture) that fixed-size
statistics compress away.

## What it is

- Architecture: 2x TransformerEncoder layers, hidden dim 64, vocab 4096
  (trigram hash tokens), sequence length 256
- Parameters: 345,665
- File: simurg_pulse.safetensors (~1.3 MB)
- Inference: MPS on Apple Silicon, CPU fallback; ~4 ms per window
- Output: probability of corruption in the current stream window

## Training data

- 40 clean live answers from the guarded endpoint (wahoo-1.5-preview via
  vLLM, Gemma-4-26B-A4B-NVFP4)
- 240 synthetic corruptions from CorruptBench (repetition loops, cross-lingual
  drift, table echo, structural garbage, semantic collapse)
- Labels are onset-aware: a window is corrupt only if its right edge is at
  least 300 chars past the true corruption onset

## Metrics

- Held-out AUROC: 0.925
- Calibration anchors stored in the safetensors metadata: lo=0.168, hi=0.367

## Usage inside SIMURG

The weights are bundled with the SIMURG package and auto-registered as the
6th detector (pulse) in the ensemble. Install with:

```
pip install simurg[deep]
```

Without torch/safetensors the detector silently contributes p=0 and the
numpy-only core behaves exactly as before.

## Standalone inference

```
from safetensors.torch import load_file
import torch

tensors = load_file("simurg_pulse.safetensors")
```

Full inference plumbing (tokenization, windowing, calibration) lives in
SIMURG under src/simurg/deep/pulse.py; the model card describes the raw
artifact.

## Citation / attribution

Trained as part of SIMURG v1.0.4 (2026-09-21) by doofzoff. MIT licensed with
the parent project.
