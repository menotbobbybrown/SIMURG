# Changelog

All notable changes to this project are documented in this file.
The format is based on Keep a Changelog.

## [1.0.4] - 2026-09-21

SIMURG Pulse — the optional deep-learning detection tier. A small streaming
transformer (345K params, 1.3 MB safetensors) trained on real answers from the
guarded endpoint plus CorruptBench corruptions. Captures SEQUENTIAL structure
(loop phase, script cadence, garbage texture) that the 15-dim statistics of
the numpy tier compress away.

### Added

- **SIMURG Pulse deep tier** (src/simurg/deep/): 2-layer transformer over the
  recent 600-char window (trigram hash tokens, 256 seq), ~345K params, ships as
  src/simurg/weights/simurg_pulse.safetensors. Trained on 40 live answers from
  wahoo-1.5-preview + 240 CorruptBench corruptions; onset-aware window labels
  mirror the numpy trainer's protocol
- **pulse detector** auto-registered in the ensemble: its calibrated
  probability joins the same conformal fusion as the five existing detectors;
  on a strong hit it cites "pulse deep-tier p=..." and attributes a taxonomy
  class
- **train_pulse.py CLI**: SIMURG_LIVE_URL/SIMURG_LIVE_MODEL env wiring, live
  sampling, BCE training with pos_weight, held-out AUROC + calibration anchors
  (lo/hi) stored in the safetensors metadata; trains on CPU in seconds (MPS
  lacks SDPA-with-dropout during training; inference uses MPS when present)
- **graceful degradation contract**: without torch / safetensors / weights the
  detector silently contributes p=0 and the numpy-only core behaves EXACTLY as
  before; install with pip install simurg[deep] to enable
- benchmark parity check: with Pulse present the CorruptBench results are
  unchanged (TPR 78/80, AUROC 0.550 on the 1-clean test split) — the deep tier
  adds recall headroom without regressing the calibrated operating point

### Changed

- pyproject: optional deep extra (torch + safetensors), pulse weights added
  to package-data

### Self-Heal repair ladder — the guard that heals

SIMURG Self-Heal — the guard that heals. A corrupt mid-stream abort is no
longer followed by a blind full retry; the diagnosis drives a targeted
continuation that stitches a clean answer in one generation's wall-clock.

#### Added

- **Self-Heal repair ladder** (GuardedLLM(heal=True), the new default):
  on a corrupt attempt the guard (1) diagnoses the fired corruption class,
  (2) trims the released prefix to its verified-clean boundary, (3) sends a
  pathology-specific continuation request ("continue from here, never repeat"
  / "stay in the original language") at slightly warmer sampling, (4) guards
  the continuation with a fresh sentinel and stitches prefix + verified tail
  into one answer with result.healed = True
- **periodic-loop onset detector** (find_loop_onset): exact loop-start
  localization via self-similarity + exact-mismatch run scan, so the whole
  degenerate loop is cut before continuation and the model never sees it
- **trim_to_clean**: verified-clean boundary extraction for aborted
  streams (structural fast path + multi-window score probes)
- **verify_final**: post-hoc full-text verification of stitched answers —
  the zero-leak guarantee now covers healed output; residue is trimmed or the
  ladder falls through to plain retries / fallback
- heal instructions per corruption class: repetition collapse, cross-lingual
  drift, structural breakdown, regurgitation; semantic collapse falls through
  to a full retry (context compromised)
- hermetic self-heal regressions (tests/test_healing.py): corrupt-then-heal
  ladder, stitched prefix/tail integrity, prefix carried into the repair
  request, zero-leak invariant

#### Changed

- GuardedLLM.chat abort handling now returns the verified released prefix of a
  corrupt attempt so the heal ladder can stitch from it (legacy abort-only
  behaviour preserved via heal=False)

## [1.0.3] - 2026-08-29

SIMURG Search — a free, hallucination-fighting web layer for AI agents. Thanks
to first-time contributor @0xsyntho for the TinyFish integration.

### Added

- **free web-search layer for agents** (`simurg.websearch`,
  `python3 -m simurg.websearch "query" [--json] [--ground]`): the TinyFish
  Search API (free, 30 req/min, $0, no card, structured
  `{title, snippet, url, site_name}` results) as a drop-in internet for local
  and small models — re-check a fact on the web before answering, or abstain on
  `no_record`. `websearch.search()` / `websearch.snippets()` for evidence,
  `websearch.ground()` for the attested | thin | no_record verdict (TinyFish +
  keyless Wikipedia cross-check). Works out of the box via a bundled free-tier
  key (Search is $0 at any wallet balance); set `TINYFISH_API_KEY` for
  dedicated limits or `TINYFISH_API_KEY=""` to opt out. Stdlib-only, zero new
  dependencies
- L4 grounded verification (Monolith) now runs on the same web layer: TinyFish
  first when a key is set, keyless DuckDuckGo scrape as the fallback; the
  verdict reports its source (`tinyfish+wiki` vs `web+wiki`)
- hermetic tests for the web layer and grounding wiring (fake Search API
  server; no network, no key needed in CI)

## [1.0.2] - 2026-08-28

### Added

- **SIMURG Monolith** — the real-time-learning + grounded-factuality layer on top
  of the base decode guard, with a Bloomberg-style terminal (`python3 -m
  simurg.veritas_dashboard`)
- online hallucination-risk model (`simurg.veritas.monolith`) that trains while it
  serves: every like / dislike is one SGD step (serving loop == training loop), so
  it adapts to your traffic with no batch-retrain gap
- white-box fact-uncertainty from decoder top-k logprobs (per-token entropy,
  margin, competing-fact detection) — a signal only a host has
- grounded verification against real evidence (Wikipedia + web): catches both a
  fabricated subject and a wrong detail on a real subject, and abstains instead of
  asserting when a claim contradicts the sources
- multilingual bootstrap dataset + trainer (EN / RU / AZ,
  `simurg.veritas.monolith_data`); shipped bootstrap model reaches held-out
  AUROC 1.0
- the terminal screenshot and a full "SIMURG Monolith" readme section with launch
  instructions and how to wire likes / dislikes from your own platform

## [1.0.1] - 2026-08-24

### Fixed

- readme images and the paper link now use absolute URLs so they render
  correctly on PyPI, not only on GitHub

### Added

- the full technical report (paper/simurg_paper.pdf) and a paper section
  in the readme
- the second author (E. Ahmadbayli) to the citation block

## [1.0.0] - 2026-08-24

### Added

first public release, published on PyPI as `simurg`.

- the zero-leak streaming guard: HOLD / RELEASE / re-check / ABORT protocol
  with conformal-calibrated thresholds and Page-Hinkley onset localization
- five-detector ensemble: char n-gram surprise, Count-Min repetition sketch,
  rolling SimHash drift, robust-z self-calibration, interpretable rules
- learned tier: 15-weight online logistic model that keeps learning in
  production via partial fit
- shipped weights and conformal thresholds, trained on real production
  traffic plus the CorruptBench synthetic benchmark
- GuardedLLM: drop-in guard for any OpenAI compatible endpoint with an
  abort, retry, fallback model ladder
- fit_custom_detector: teach SIMURG a new failure mode from your own examples,
  with an honesty gate that reports NOT DETECTABLE instead of a false promise
- CorruptBench synthetic dataset builder and a reproducible end-to-end
  benchmark (python3 -m simurg.data.evaluate, seed 7)
- live training dashboard with per epoch weight evolution
- live guard dashboard: SSE server, real time score and 15 feature charts,
  session recording, replay at up to 128x for postmortems
- regression tests: sentinel behavior and end-to-end dashboard tests
  against a mock OpenAI compatible upstream
