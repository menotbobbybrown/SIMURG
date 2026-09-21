# ═══════════════════════════════════════════════════════════════════════════════
# SIMURG · Streaming Integrity Monitor & Universal Regeneration Guard
#
# Developed by doofZ (a.k.a Farid Aghayev from HAL-X AI)
# Co-Founder & Head of AI at HAL-X AI.
#
# openai_guard — drop-in SIMURG protection for ANY OpenAI-compatible endpoint
# (vLLM, llama.cpp server, TGI, Ollama, OpenAI/OpenRouter, ...). Wraps a streaming
# /v1/chat/completions call in the zero-leak protocol and an abort -> heal ->
# retry -> fallback ladder, using nothing but the standard library.
#
# SELF-HEALING (new in 1.0.4)
# ---------------------------
# A blind retry regenerates the WHOLE answer and often corrupts again --
# repetition collapse in particular is near-deterministic under the same
# sampling. With heal=True (default), a mid-stream abort is treated as a
# diagnosis: the Healer maps the fired corruption class onto a targeted
# continuation request (continue from here, never repeat), the continuation
# is guarded by a fresh sentinel, and the verified prefix + verified tail are
# stitched into ONE clean answer. Wall-clock stays close to a single
# generation instead of two. If the heal itself corrupts, the ladder falls
# through to plain retries and the fallback model, exactly as before.
#
# fallback accepts another GuardedLLM config (different endpoint/model), giving
# the same primary -> retry -> fallback-model ladder we run in production.
# ═══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..learning.model import OnlineLogReg
from ..detection.sentinel import CORRUPT, Simurg
from ..healing import (Healer, MIN_PREFIX_CHARS, REPAIR,
                       trim_to_clean, verify_final)

# Extra sampling used ONLY for heal-continuation requests. Degenerate loops are
# nearly deterministic under greedy decoding, so the continuation runs with a
# slightly warmer temperature. User-supplied sampling always wins.
_HEAL_SAMPLING = {"temperature": 1.0, "top_p": 0.95}


@dataclass
class Attempt:
    label: str                 # "primary" | "heal-1" | "retry-1" | ... | "fallback"
    state: str                 # clean | suspect | corrupt | error
    reasons: List[str] = field(default_factory=list)
    onset_char: Optional[int] = None
    chars: int = 0
    strategy: str = ""         # "repair" for heal continuations, "" otherwise


@dataclass
class GuardResult:
    text: str                  # the accepted (clean) text, "" if every rung failed
    ok: bool
    verdict: str               # final sentinel state of the accepted attempt
    attempts: List[Attempt] = field(default_factory=list)
    healed: bool = False       # True when the answer was stitched from a repair

    @property
    def recovered(self) -> bool:
        return self.ok and len(self.attempts) > 1


def _heal_messages(messages: list, prefix: str, instruction: str) -> list:
    """Context for a targeted continuation: the verified prefix as an assistant
    turn, followed by the pathology-specific steering instruction."""
    msgs = list(messages)
    if msgs and msgs[-1]["role"] == "assistant":
        msgs = msgs[:-1]
    msgs.append({"role": "assistant", "content": prefix})
    msgs.append({"role": "user", "content": instruction})
    return msgs


class GuardedLLM:
    """An OpenAI-compatible chat client whose streams are guarded by SIMURG.

    heal=True (default) enables the Self-Heal ladder: a corrupt attempt is
    repaired by a targeted continuation from its clean prefix instead of a
    blind full retry. Disable with heal=False for legacy abort-only behaviour.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 retries: int = 1, fallback: "GuardedLLM | None" = None,
                 simurg_model_path: str | None = None,
                 expected_scripts=("latin", "cyrillic"),
                 request_extra: dict | None = None, timeout: float = 300.0,
                 heal: bool = True):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.retries = max(0, retries)
        self.fallback = fallback
        self.heal = heal
        self.expected_scripts = tuple(expected_scripts)
        self.request_extra = request_extra or {}
        self.timeout = timeout
        self.healer = Healer()
        self._learned = None
        if simurg_model_path:
            try:
                self._learned = OnlineLogReg.load(simurg_model_path)
            except Exception:
                self._learned = None

    # ── one guarded streaming attempt ────────────────────────────────────────
    def _attempt(self, label: str, messages: list, on_token,
                 **gen_kwargs) -> tuple[Attempt, str]:
        sentinel = Simurg(expected_scripts=self.expected_scripts, model=self._learned)
        body = {"model": self.model, "messages": messages, "stream": True,
                **self.request_extra, **gen_kwargs}
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"}, method="POST")
        accepted: List[str] = []
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                buf = b""
                done = False
                while not done:
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            done = True
                            break
                        try:
                            delta = json.loads(payload)["choices"][0]["delta"]
                        except Exception:
                            continue
                        token = delta.get("content")
                        if not token:
                            continue
                        v = sentinel.feed(token)
                        if v.state == CORRUPT:
                            # zero-leak abort: close the HTTP stream immediately.
                            # The RELEASED prefix (verified clean) is returned so
                            # the heal ladder can stitch from it.
                            return (Attempt(label, "corrupt", list(v.reasons),
                                            v.onset_char, sentinel.f.total_len),
                                    "".join(accepted))
                        if v.released:
                            accepted.append(v.released)
                            if on_token:
                                on_token(v.released)
        except Exception as e:
            return Attempt(label, "error", [str(e)[:200]], None,
                           sentinel.f.total_len), "".join(accepted)
        v = sentinel.finish()
        if v.state == CORRUPT:
            return (Attempt(label, "corrupt", list(v.reasons), v.onset_char,
                            sentinel.f.total_len), "".join(accepted))
        if v.released:
            accepted.append(v.released)
            if on_token:
                on_token(v.released)
        return (Attempt(label, v.state, list(v.reasons), None,
                        sentinel.f.total_len), "".join(accepted))

    # ── the ladder ───────────────────────────────────────────────────────────
    def chat(self, messages: list, on_token: Callable[[str], None] | None = None,
             **gen_kwargs) -> GuardResult:
        """Guarded completion with the abort -> heal -> retry -> fallback ladder.
        on_token receives only SENTINEL-RELEASED text (zero-leak): on a corrupt
        attempt that was held, nothing is ever forwarded; on a mid-stream abort,
        the host UI should replace the shown text with the next attempt's output.

        With healing enabled, a mid-stream abort triggers a targeted continuation
        from the verified clean prefix (see module docstring); the result is
        stitched and returned as one clean answer with healed=True."""
        attempts: List[Attempt] = []
        rungs = [("primary", self)] + [(f"retry-{i + 1}", self) for i in range(self.retries)]
        heal_count = 0
        for label, cl in rungs:
            att, text = cl._attempt(label, messages, on_token, **gen_kwargs)
            attempts.append(att)
            if att.state in ("clean", "suspect"):
                return GuardResult(text, True, att.state, attempts)
            if self.heal and att.state == CORRUPT:
                # The released prefix may contain a corrupt tail (text emitted
                # between the last clean checkpoint and the abort). Trim it down
                # to the verified-clean boundary before continuing from it.
                # trim_to_clean returns None when even the tail window cannot be
                # made clean -- the context is contaminated, so fall through to
                # the plain retry ladder instead of continuing from it.
                clean_prefix = trim_to_clean(text, self.expected_scripts)
                plan = self.healer.plan(att)
                if (plan.strategy == REPAIR and clean_prefix
                        and len(clean_prefix) >= MIN_PREFIX_CHARS
                        and heal_count < self.retries + 1):
                    heal_count += 1
                    hlabel = f"heal-{heal_count}"
                    healed_msgs = _heal_messages(messages, clean_prefix, plan.instruction)
                    heal_kwargs = dict(_HEAL_SAMPLING, **gen_kwargs)
                    hatt, htext = cl._attempt(hlabel, healed_msgs, on_token,
                                              **heal_kwargs)
                    hatt.strategy = plan.strategy
                    attempts.append(hatt)
                    if hatt.state in ("clean", "suspect"):
                        stitched = self.healer.stitch(clean_prefix, htext)
                        # POST-HOC VERIFICATION: the zero-leak guarantee for
                        # healed answers -- whatever is returned must itself
                        # score clean end to end. If a degenerate residue
                        # survived the seam, trim it; if nothing recoverable
                        # remains, fall through to the next ladder rung.
                        if not verify_final(stitched, self.expected_scripts):
                            trimmed = trim_to_clean(stitched,
                                                    self.expected_scripts)
                            if (trimmed and len(trimmed) >= MIN_PREFIX_CHARS
                                    and verify_final(trimmed,
                                                     self.expected_scripts)):
                                stitched = trimmed
                            else:
                                continue
                        return GuardResult(stitched, True, hatt.state, attempts,
                                           healed=True)
        if self.fallback is not None:
            att, text = self.fallback._attempt("fallback", messages, on_token,
                                               **gen_kwargs)
            attempts.append(att)
            if att.state in ("clean", "suspect"):
                return GuardResult(text, True, att.state, attempts)
        return GuardResult("", False, "corrupt", attempts)
