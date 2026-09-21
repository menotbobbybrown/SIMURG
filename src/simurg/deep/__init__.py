# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# deep — SIMURG Pulse, the optional deep-learning detection tier. A small
# streaming transformer (~350K params, safetensors) trained on live answers
# from the guarded endpoint plus CorruptBench corruptions. See pulse.py
# (model + inference), train_pulse.py (trainer CLI), pulse_detector.py
# (ensemble wiring). Importing this package never imports torch; torch is
# loaded lazily at evaluate-time when weights are present.
# ═══════════════════════════════════════════════════════════════════════════════
from .pulse_detector import PulseDetector

__all__ = ["PulseDetector"]
