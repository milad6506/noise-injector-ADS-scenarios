# Fault-Injection Taxonomy for ADS Scenario Data

The catalog of everything `noise-injector` injects — now or in later rounds. This document is the
spec the implementation is generated from; each mode carries an ID, default temporal pattern,
severity curve, and applicability notes.

## 0. Model

Every injection is a 5-tuple:

```
Injection = (mode, scope, pattern, magnitude, rate) + seed
```

- **scope** — where it lands: `scenario | camera(i) | frame(t) | object(o) | qa(q) | harness-file`
- **pattern** — temporal shape over the scenario's T frames:
  | Pattern | Meaning | Example |
  |---|---|---|
  | persistent | all frames, whole scenario | fog, low_light, jpeg |
  | intermittent | Markov on/off flicker (p_on grows with severity) | rain gusts, flicker |
  | burst | contiguous k frames (k scales with severity) | frame_freeze, motion_blur during jerk |
  | transient | exactly 1 frame | glare flash, phantom object |
  | camera_scoped | 1 of N cameras, all frames | camera_dropout, dirty lens |
- **magnitude** — per-family severity curve (1–5)
- **rate** — probability of discrete errors (Layers B/C)
- **seed** — `sha256(scenario_id ‖ recipe_id)` → byte-identical offspring across runs and models

## 1. Layer A — Sensor / image corruptions (implemented)

| ID | Family | Default pattern | Severity curve (S1→S5) |
|---|---|---|---|
| A-01 | fog | persistent | veil 0.15→0.75, contrast −40% |
| A-02 | rain | persistent | 40→400 streaks, darken 5→25% |
| A-03 | snow | persistent | particle density 0.2→2% |
| A-04 | water_splash | persistent | blob alpha 0.15→0.45 |
| A-05 | low_light | persistent | gamma 1.4→3.2 + read noise |
| A-06 | overexposure | persistent | highlight blowout 0.15→0.46 |
| A-07 | glare | **transient** | radial flare 0.15→0.5 |
| A-08 | bloom | persistent | glow strength 0.2→0.6 |
| A-09 | motion_blur | persistent | kernel 5→33 px |
| A-10 | defocus | persistent | σ 0.6→3.5 |
| A-11 | rolling_shutter | persistent | row shear 1→8 px |
| A-12 | awgn | persistent | σ 2→22 |
| A-13 | salt_pepper | persistent | p 0.2→3.2% |
| A-14 | dead_pixels | persistent | 1→12 clusters |
| A-15 | banding | persistent | amplitude 3→16 |
| A-16 | chromatic_aberration | persistent | R/B shift 1→7 px |
| A-17 | lens_droplets | persistent | 3→24 droplets |
| A-18 | lens_dirt | persistent | smudge coverage 2→11% |
| A-19 | color_cast | persistent | channel gain 5→27% |
| A-20 | quantization | persistent | color levels 48→3 |
| A-21 | jpeg | persistent | quality 55→5 |
| A-22 | frame_freeze | **burst** | copies previous frame, burst ≤ severity |
| A-23 | camera_dropout | **camera_scoped** | black (S1–3) / missing (S4–5) |
| A-24 | camera_desync | **transient** | one camera stale by 1 frame |

Scene-weather (a *semantic* world condition the ADS must detect and respond to — ODD fallback)
is distinct from sensor-weather (an *image artifact* where the ground truth is unchanged). This
tool implements sensor-weather; scene-weather belongs to scenario content, not injection.

## 2. Layer B — Annotation / perception errors (implemented)

Error placement is structured, not uniform: confusions follow semantic sets
(Vehicle↔Cyclist↔Pedestrian), misses weight toward small/occluded objects.

| ID | Family | Default pattern | Rate/param (S1→S5) |
|---|---|---|---|
| B-01 | missed_detections | persistent | 5→50% of objects |
| B-02 | phantom_objects | persistent | 3→25% of objects added |
| B-03 | bbox_jitter | persistent | σ 4→34 px |
| B-04 | class_confusion | persistent | 2→30% |
| B-05 | status_flip | persistent | 2→30% (Moving↔Stationary) |
| B-06 | max_det_truncation | persistent | cap 10→2 objects/frame |
| B-07 | llm_description_drift | persistent | 2→30%, LLM-rewritten with one wrong attribute (bring-your-own backend) |

Signal-state flip (traffic-light color, flasher on/off) is the highest-value B-mode for
expectation-based evaluation (≈40 of the 205 requirement families hinge on light/signal state
reading) — specialized in a later round on datasets carrying signal states.

## 3. Layer C — QA / training-data errors (implemented)

| ID | Family | Default pattern | Rate (S1→S5) |
|---|---|---|---|
| C-01 | qa_wrong_answer | persistent | 2→30% |
| C-02 | qa_answer_drift | persistent | 2→25% (left/right, stop/go, red/green swaps) |
| C-03 | qa_category_misassign | persistent | 2→25% |
| C-04 | qa_drop | persistent | 2→30% |
| C-05 | qa_duplicate | persistent | 2→25% |
| C-06 | qa_crossref_break | persistent | 2→30% (`<c#>` refs point wrong) |
| C-07 | llm_qa_rewrite | persistent | 2→30%, LLM-rewritten answers — styles: plausible_wrong / vague / hallucinated (bring-your-own backend) |

LLM-powered modes (B-07, C-07) execute through a pluggable OpenAI-compatible backend supplied
at run time (`--llm-base-url` / `--llm-model`, key via env var). Disk-cached responses keep
re-renders deterministic.

Three error regimes drive rate presets:

| Regime | Object-label errors | QA answer errors | Character |
|---|---|---|---|
| benign human | 3–8% | 5–10% (15% prediction) | semantic confusion, small-object misses |
| systematic pipeline | 20–50% dropout far/occluded, 5–15% phantom | graph gaps | what auto-labeled sets contain |
| adversarial | 0.1–5% budget, targeted | safety-critical only | see Layer E |

## 4. Layer D — Systemic faults (spec-only, next round)

ISO 26262/21448-flavoured input/output faults: sensor stuck-at (frozen value stream),
out-of-range values, processing latency (frame delayed by k timestamps), calibration drift
(geometric distortion slowly changing), timestamp skew (cross-sensor misalignment), sync fault.
These operate on the *scenario stream* rather than single frames and need a scenario format
carrying absolute timestamps — the Waymo adapter (timestamped) will be the first carrier.

## 5. Layer E — Adversarial data corruption (spec-only, next round)

1. **Untargeted label noise** — 10–20% random flips (breaks training)
2. **Targeted safety-critical flips** — 1% budget, only planning/behavior answers, flipped toward
   the majority class ("going ahead") — exploits dataset imbalance
3. **Clean-label poisoning** — mismatched image↔annotation pairs, 0.5–2% of training set
4. **Backdoor trigger** — patched frames (1–3%) → pinned unsafe behavior answer; measures
   whether fine-tuning learns hidden shortcuts
5. **Eval-time** (not training): PGD/adversarial patches — out of v1 scope

## 6. Layer F — Agent-harness faults (implemented)

Faults injected into harness *artifacts* (prompts, configs) — never executes models; originals
are never modified (corruption lands on a copied tree + manifest).

| ID | Op | Effect |
|---|---|---|
| F-01 | prompt_truncation | file cut to 90→30% length |
| F-02 | prompt_section_delete | 1–3 paragraphs dropped |
| F-03 | prompt_instruction_injection | adversarial instruction appended |
| F-04 | prompt_garble | 2→20% alphanumeric chars replaced |
| F-05 | prompt_format_break | code fences / braces / headers stripped, lines dropped |
| F-06 | vocab_drop | detector classes removed |
| F-07 | vocab_misspell | class names corrupted (semantic aliases lost) |
| F-08 | vocab_swap | class replaced with sibling (car↔van) |
| F-09 | confidence_drift | detector threshold +0.05→0.4 |
| F-10 | max_tokens_starve | generation budget ÷2 → ÷32 |
| F-11 | quality_drift | max_px 512→128, jpeg quality 60→10 |
| F-12 | temperature_drift | +0.2→+1.0 |

## 7. Reproducibility contract

- Seed per (scenario, recipe) via `sha256`; per-corruption/per-frame child seeds derived
  deterministically — shared transient patterns across all models by construction.
- `manifest.json` per offspring: recipe, seeds, every injected event with per-frame targets and
  params. Same inputs → byte-identical output (verified in tests).
- Scoring consumers can subtract *planted* errors (manifest) from model errors.

## 8. Combinatorics guidance

A 4-frame × 6-cam scenario with 47 modes × 5 severities → up to ~236 children per scenario.
Materialize offline only curated eval subsets; generate the rest on-the-fly at training/eval
time. Compound recipes (multiple ops in one recipe) are first-class; the combinatorial pairing
matrix (600 expectation scenarios × noise children) should be sampled, not enumerated.

## 9. References

- Xie et al., *Are VLMs Ready for Autonomous Driving?* (DriveBench, ICCV 2025) — 17 settings,
  corruption families (weather/external/sensor/motion/transmission), VLM hallucination under
  missing/degraded visual input
- Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions* (ImageNet-C)
- Michaelis et al., *Benchmarking Robustness in Object Detection* (COCO-C)
- Northcutt et al., *Pervasive Label Errors in Test Sets Destabilize ML Benchmarks* (~6% label error)
- ISO 21448 (SOTIF), ISO 26262 — triggering conditions, fault classes (permanent/intermittent/transient)
