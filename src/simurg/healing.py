# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# healing — the Self-Heal repair policy. When the zero-leak guard aborts a
# stream mid-flight, a blind retry regenerates the whole answer and often
# corrupts AGAIN (repetition collapse is near-deterministic: same seed, same
# degenerate attractor). The healer instead maps the diagnosed corruption
# class onto a TARGETED treatment:
#
#   repetition loop   -> CONTINUE from the clean prefix, "never repeat"
#   script drift      -> CONTINUE, "stay in the original language"
#   structural garbage-> CONTINUE, "ignore malformed characters"
#   regurgitation     -> CONTINUE, "answer the question, quote nothing"
#   semantic collapse -> full RETRY (the context itself is compromised)
#
# The continuation stream is itself guarded by a fresh sentinel, so a heal that
# re-corrupts falls through to the plain retry ladder. The host stitches the
# verified prefix + verified tail into one clean answer — same wall-clock as a
# single generation instead of two.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

from dataclasses import dataclass, field

from .core import DRIFT, REGURGITATION, REPETITION, SEMANTIC, STRUCTURAL
from .detection.sentinel import CORRUPT, Simurg

REPAIR = "repair"   # continue from the clean prefix and stitch
RETRY = "retry"     # context is compromised; regenerate from scratch

# Minimum clean prefix needed before stitching is worth it; below this a plain
# retry is cheaper than a heal that re-sends a near-empty context.
MIN_PREFIX_CHARS = 160

_INSTRUCTIONS = {
    REPETITION: ("Continue your answer EXACTLY from where it stopped. "
                 "Never repeat any sentence, phrase, or list item you already "
                 "wrote above. Keep the same language and style."),
    DRIFT: ("Continue your answer in its ORIGINAL language and script. "
            "Do not switch languages mid-sentence and do not translate."),
    STRUCTURAL: ("Continue your answer. Ignore any malformed or garbled "
                 "characters that may have appeared above and keep writing "
                 "clean, well-formed text."),
    REGURGITATION: ("Continue answering the user's original question directly. "
                    "Do not quote, paste, or echo documents, code, or "
                    "boilerplate."),
}


@dataclass
class RepairPlan:
    strategy: str                        # REPAIR or RETRY
    classes: tuple = ()
    reasons: list = field(default_factory=list)
    onset_char: int | None = None
    instruction: str = ""


def _infer_classes(reasons, classes) -> set:
    cls = set(classes or [])
    text = " ".join(reasons or []).lower()
    if "repetition" in text or "vocabulary collapse" in text:
        cls.add(REPETITION)
    if "foreign" in text:
        cls.add(DRIFT)
    if ("structural" in text or "numeric dump" in text
            or "same-char run" in text):
        cls.add(STRUCTURAL)
    if "regurgitat" in text:
        cls.add(REGURGITATION)
    if "semantic" in text or "simhash" in text:
        cls.add(SEMANTIC)
    return cls


class Healer:
    """Maps a corrupt verdict onto a targeted repair plan.

    plan(verdict) takes any object exposing .reasons, .classes, .onset_char
    (e.g. a sentinel Verdict or a guard Attempt)."""

    def plan(self, verdict) -> RepairPlan:
        reasons = list(getattr(verdict, "reasons", []) or [])
        classes = list(getattr(verdict, "classes", []) or [])
        onset = getattr(verdict, "onset_char", None)
        cls = _infer_classes(reasons, classes)

        # Semantic/topic collapse compromises the context itself — continuing
        # would carry the derailment forward. Every other class is localizable:
        # the clean prefix is still valid and the model can be steered around
        # the specific pathology.
        strategy = RETRY if SEMANTIC in cls else REPAIR
        instruction = ""
        if strategy == REPAIR:
            for c in (REPETITION, DRIFT, STRUCTURAL, REGURGITATION):
                if c in cls:
                    instruction = _INSTRUCTIONS[c]
                    break
            if not instruction:
                instruction = _INSTRUCTIONS[REPETITION]
        return RepairPlan(strategy, tuple(sorted(cls)), reasons, onset, instruction)

    def stitch(self, prefix: str, continuation: str) -> str:
        """Join prefix + continuation, removing any overlap the model re-echoed
        at the seam (models often re-output the last sentence they see)."""
        return strip_overlap(prefix, continuation)


def _snap_forward(text: str, cut: int, min_len: int) -> str:
    if cut < min_len:
        return ""
    nxt = text.find(" ", cut)
    if cut < len(text) and 0 <= nxt - cut < 200:
        cut = nxt + 1                          # snap forward to next whitespace
    return text[:cut]


def verify_final(text: str, expected_scripts=("latin", "cyrillic")) -> bool:
    """Post-hoc full-text verification of a stitched answer.

    The zero-leak guarantee for healed answers: whatever is returned to the
    user must itself score clean end to end. One O(n) pass through a fresh
    sentinel; hundreds of kchars/sec, so a 4k-char answer costs ~20 ms.
    """
    if not text:
        return False
    s = Simurg(expected_scripts=expected_scripts)
    step = 256
    for i in range(0, len(text), step):
        s.feed(text[i:i + step])
    return s.finish().state != CORRUPT


def find_loop_onset(text: str, tail_len: int = 800,
                    min_period: int = 8, max_period: int = 240,
                    min_agree: float = 0.85) -> int | None:
    """Locate the START of a repetition collapse inside text, or None.

    Degenerate loops are near-periodic: the model locks onto a unit of
    length p and re-emits it. This detector (1) finds the dominant period p
    of the recent tail by self-similarity, and (2) walks back to the earliest
    position where at least two full periods match exactly -- that position
    is the loop onset. A guardrail requires the periodic run to cover at
    least 60% of the last 600 chars, so legitimate prose with an occasional
    repeated phrase is never trimmed. O(n).
    """
    n = len(text)
    if n < min_period * 3:
        return None
    t_start = max(0, n - tail_len)
    tail = text[t_start:]
    best_p, best_agree = None, 0.0
    for p in range(min_period, max_period + 1):
        m = len(tail) - p
        if m < min_period:
            break
        agree = 0
        for i in range(m):
            agree += tail[i] == tail[i + p]
        agree /= m
        if agree > best_agree:
            best_p, best_agree = p, agree
    if best_p is None or best_agree < min_agree:
        return None
    p = best_p
    # exact-mismatch run scan: find the last maximal run of positions i with
    # text[i] == text[i + p]; two consecutive full periods must agree exactly
    need = 2 * p
    run_start = None
    run = 0
    for i in range(n - p):
        if text[i] == text[i + p]:
            run += 1
        else:
            if run >= need:
                run_start = i - run
            run = 0
    if run >= need:
        run_start = n - p - run
    if run_start is None:
        return None
    run_len = (n - p) - run_start
    guard = 0.6 * min(600, n)
    if run_len < guard:
        return None
    return run_start


def trim_to_clean(text: str, expected_scripts=("latin", "cyrillic"),
                  min_len: int = 0, window: int = 1200) -> str | None:
    """Return the longest verified-clean prefix of text, or None when the
    text cannot be trusted as continuation context.

    The released prefix of an aborted stream can contain a corrupt tail
    (text emitted between the last clean checkpoint and the abort). Cutting
    works in two stages:

    1. structural loop detection finds the exact onset of a periodic collapse
       and trims the WHOLE loop (score-based probes see a diluted window and
       would under-cut);
    2. a score-based binary search handles non-periodic contamination.

    Each probe re-feeds the last window chars ending at the candidate cut
    through a fresh Simurg at full speed (hundreds of kchars/sec), so even a
    10k-char answer costs less than 1 ms. Returns None when even the tail
    window cannot be made clean -- in that case the context is contaminated
    and the host should regenerate from scratch instead of continuing.
    """
    if not text:
        return None

    def clean_cut(at: int) -> bool:
        if at <= 0:
            return True
        # Probe with SEVERAL window lengths ending at the cut: a short window
        # sees local degeneration that a long window dilutes below threshold.
        # The cut is clean only if every window is clean.
        for w in (250, 600, window):
            start = max(0, at - w)
            if at - start < 60:
                continue
            s = Simurg(expected_scripts=expected_scripts)
            t = text[start:at]
            step = max(1, len(t) // 40)
            for i in range(0, len(t), step):
                s.feed(t[i:i + step])
            # Compare the fused score directly against the corrupt threshold:
            # finish() applies the 2-hit hysteresis of the live protocol, which
            # would let a corrupt tail pass a single-shot probe.
            fused, _hard = s._score()
            if fused.p >= s.fusion.corrupt_at:
                return False
        return True

    # Fast path: structural loop detection finds the exact onset and trims the
    # WHOLE loop before any score probing.
    onset = find_loop_onset(text)
    if onset is not None:
        cut = onset
        if clean_cut(cut):
            return _snap_forward(text, cut, min_len)
        # loop onset itself sits in a contaminated zone -> not recoverable
        return None

    if clean_cut(len(text)):
        return text
    lo, hi = 0, len(text)
    while hi - lo > 100:                       # coarse binary search
        mid = (lo + hi) // 2
        if clean_cut(mid):
            lo = mid
        else:
            hi = mid
    cut = (hi // 100) * 100                    # snap to a 100-char boundary
    if cut < min_len or not clean_cut(cut):
        return None
    return _snap_forward(text, cut, min_len)


def strip_overlap(prefix: str, continuation: str, max_scan: int = 240) -> str:
    """If the continuation starts by re-echoing the end of the prefix, drop the
    echoed span. Case/whitespace-insensitive matching; bounded scan cost."""
    if not prefix or not continuation:
        return prefix + continuation
    n = min(len(prefix), max_scan)
    for i in range(0, n, 4):                      # 4-char stride: good enough, cheap
        tail = prefix[-(n - i):]
        if len(tail) < 24:
            break
        if _insensitive_startswith(continuation, tail):
            return prefix + continuation[len(tail):]
    return prefix + continuation


def _insensitive_startswith(a: str, b: str) -> bool:
    return a.lower()[:len(b)].replace(" ", "") == b.lower().replace(" ", "")[:len(b)]
