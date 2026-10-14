# noise-injector

Deterministic fault-injection tool for autonomous-driving research: point it at a scenario
dataset, and it creates **offspring scenarios** — the same scenario with controlled noise,
errors, or fault patterns injected — each with a full provenance manifest.

Built to study how perception models (object detectors) and driving VLMs behave under
different levels and shapes of sensory degradation, data errors, and harness faults.

## Concept

```
parent scenario (4 key frames × 6 cameras + QA graph)
        │
        │  recipe = (family × severity × pattern) + deterministic seed
        ▼
offspring scenarios ──► images/ + annotations.json + manifest.json
```

- **Family** — what is injected (fog, AWGN, missed detections, QA corruption, prompt faults…)
- **Severity** — 1–5 intensity curve per family (ImageNet-C convention)
- **Pattern** — temporal shape: `persistent` (all frames), `intermittent` (Markov on/off),
  `burst` (contiguous frames), `transient` (single frame), `camera_scoped` (one camera, all frames)
- **Determinism** — `seed = sha256(scenario_id ‖ recipe_id)`; same inputs → byte-identical
  offspring, so every model under test sees exactly the same corrupted inputs.

## Layers

| Layer | Target | Families |
|---|---|---|
| **A** | camera images | fog, rain, snow, water_splash, low_light, overexposure, glare, bloom, motion_blur, defocus, rolling_shutter, awgn, salt_pepper, dead_pixels, banding, chromatic_aberration, lens_droplets, lens_dirt, color_cast, quantization, jpeg + structural: frame_freeze, camera_dropout, camera_desync |
| **B** | annotations (`key_object_infos`) | missed_detections, phantom_objects, bbox_jitter, class_confusion, status_flip, max_det_truncation, llm_description_drift |
| **C** | QA graph data | qa_wrong_answer, qa_answer_drift, qa_category_misassign, qa_drop, qa_duplicate, qa_crossref_break, llm_qa_rewrite |
| **F** | agent-harness artifacts (prompts, config.json) — never executes models | prompt_truncation, prompt_section_delete, prompt_instruction_injection, prompt_garble, prompt_format_break, vocab_drop, vocab_misspell, vocab_swap, confidence_drift, max_tokens_starve, quality_drift, temperature_drift |

Layer D (systemic faults: latency, stuck-at, out-of-range, timestamp skew) and Layer E
(adversarial: poisoning, backdoors, targeted label flips) are specified in
[docs/TAXONOMY.md](docs/TAXONOMY.md) — implementation planned for the next round.

**Deep-dive**: [docs/noise_types.html](docs/noise_types.html) — the philosophy and exact
algorithm behind every noise family (open in a browser).

## Supported dataset adapters

| Type | Layout |
|---|---|
| `nuscenes` | DriveLM-nuScenes root: `v1_1_train_nus.json` + `nuscenes/samples/{CAM}/*.jpg` |
| `carla` | DriveLM-CARLA root: `drivelm_carla_vqas/{Town}/{Scenario}/{Route}/*.json` + `PDM_Lite_Carla_LB2/...` |
| `waymo` | `camera_box/*.parquet` (Waymo Open Dataset v2.x). Annotation-only scenarios (2D boxes, no imagery) — image families skip automatically |

The scenario model (`Scenario` / `Frame` / `CameraImage`) is adapter-based — adding a dataset
is one small file.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# severity sweep for one family on the first DriveLM-nuScenes scenario
noise-injector inject --type nuscenes --root /path/to/DriveLM \
  --family fog --severities 1-5 --scenarios 1 --out offspring/

# compound recipe on one CARLA route
noise-injector inject --type carla --root /path/to/DriveLM \
  --recipe recipes/sensor_stress.yaml --scenarios Town01/ControlLoss/Route0_Rep0 --out offspring/

# annotation noise on a Waymo segment (no imagery needed)
noise-injector inject --type waymo --root /path/to/waymo_sample \
  --family missed_detections --severities 1-5 --scenarios <segment_id> --out offspring/

# corrupt a *copy* of an agent harness (prompts + config; originals untouched)
noise-injector harness --root /path/to/judge_agent2 \
  --recipe recipes/harness_faults.yaml --out offspring/harness_demo

noise-injector list-families
noise-injector preview --type nuscenes --root ... --family fog --severities 1-5 --out fog.png

# full resilience experiment: LLM-managed rounds against your target model
noise-injector run --experiment experiments/pilot.yaml --out runs/pilot
```

## Resilience experiments (rounds of perturbations)

`noise-injector run` executes the complete loop — see [`experiments/pilot.yaml`](experiments/pilot.yaml):

```
[round 0] clean baseline            → score the targets on unmodified scenarios
[round 1..N]                        → the LLM engine proposes each round's perturbations
    engine proposal (JSON policy)   → validated against the family catalog (fallback: rotation sweep)
    inject                          → offspring scenarios rendered (deterministic seeds)
    evaluate                        → every target scores every offspring
    engine sees results             → escalates severities / probes untested families
[final]                             → resilience = mean retention (noisy/clean) per target → HTML report
```

- **engine** — the LLM that *manages the process*: any OpenAI-compatible endpoint (key via env var only)
- **dataset** — training/scenario dataset link (`nuscenes | carla | waymo` adapter)
- **targets** — the models under test, e.g. the model that runs on the device:
  - `type: vlm` — any OpenAI-compatible vision endpoint (ollama / llama.cpp server / cloud);
    `role: driving` (per-frame ego decisions, judge-scored by the engine) or
    `role: perception` (object presence, scored objectively against key-object GT)
  - `type: yolo` — local open-vocabulary detector (optional `ultralytics` dependency)
- **scoring** — `recall_only` (default: fair to key-object GT; FP rate reported as diagnostic) or `strict` (recall − FP)
- runs are **resumable** (`state.json`); the report (`resilience_report.html`) contains the final
  resilience table, retention per (family, severity), and every engine rationale

Offspring `annotations.json` preserves the DriveLM schema (per-frame `key_object_infos`,
`QA`, `image_paths`), so downstream consumers (e.g. judge/eval harnesses) can ingest corrupted
bundles the same way as parent scenarios. The `manifest.json` records every injected event
(family, params, pattern, per-frame targets) — planted errors are distinguishable from model
errors when scoring.

## Output structure

```
offspring/nuscenes/<scene_id>/fog_s3/
  images/{frame_id}__{CAM}.jpg
  annotations.json
  manifest.json
```

## Recipes

Recipe files (YAML) combine one or more ops:

```yaml
name: sensor_stress_compound
ops:
  - family: low_light
    severity: 3
  - family: awgn
    severity: 2
  - family: frame_freeze
    severity: 2
```

Harness ops support a `targets:` list to scope which files are affected.

## LLM backend (bring your own API)

Layer B/C corruptions can be executed by **an LLM you connect at run time** — any
OpenAI-compatible endpoint (OpenAI, OpenRouter, Together, vLLM, ollama, …):

```bash
# cloud endpoint — key read from env var, never passed on the CLI, never stored
export NOISE_INJECTOR_API_KEY=sk-...
noise-injector inject --type nuscenes --root /path/to/DriveLM \
  --family llm_qa_rewrite --severities 3 \
  --llm-base-url https://api.openai.com/v1 --llm-model gpt-4o-mini --out offspring/

# local ollama — no key needed
noise-injector inject --type nuscenes --root /path/to/DriveLM \
  --family llm_qa_rewrite --severities 3 \
  --llm-base-url http://127.0.0.1:11434/v1 --llm-model llama3.2:3b --out offspring/
```

LLM ops available: `llm_qa_rewrite` (styles: `plausible_wrong`, `vague`, `hallucinated` —
set via op `params`), `llm_description_drift`. Responses are cached on disk
(`.llm_cache.json`, gitignored) so re-renders stay deterministic; the manifest records the
backend model name but never the key.

## Privacy / secrets policy

- API keys only ever come from **environment variables** — the tool never accepts, logs, or
  stores a key argument; `.env` files are gitignored
- Manifests **redact dataset root paths by default** (`<dataset-root>`)
- The report generator scrubs path-like strings from rendered output
- Test suite runs fully offline (fake LLM stub; synthetic fixtures)

## Tests

```bash
pytest
```

The suite runs on synthetic fixtures — no dataset download required.

## Examples

Ready-to-run scripts live in [`examples/`](examples/):

| Script | What it does |
|---|---|
| `examples/waymo_samples.sh` | fetch Waymo camera_box segments (gsutil) → inject annotation noise → HTML report with the scenario tree |
| `examples/carla_public.sh` | fetch a public DriveLM-CARLA scene package (181 MB, no token) → inject fog → report |
| `examples/harness_from_link.sh` | fetch an agent harness from a git/repo link → corrupt prompts/config copy |

The typical one-call flow — name a dataset, a noise class, severities, and a sample size; the tool
fetches, perturbs, and writes `report.html` with the **parent → children scenario tree**:

```bash
noise-injector generate --dataset waymo  --samples 3 --family missed_detections --severities 1-3 --out testset/
noise-injector generate --dataset carla  --scenario ControlLoss --samples 1 --family fog --severities 1-3 --out testset/
noise-injector generate --dataset drivelm --split val --samples 2 --family qa_wrong_answer --severities 2 --out testset/
```

Datasets: `waymo` (gsutil-authed GCS), `carla` (public HF packages), `drivelm` (gated HF repo — set
`HF_TOKEN`; `--with-images` also fetches the large image archive), `lyft` (no longer hosted — use
`fetch --link <url>` for any direct package URL). Harness injection also accepts a **link**:

```bash
noise-injector harness --link https://github.com/you/your-agent-harness --op prompt_garble --out corrupted_copy/
```

## Acknowledgments

This tool and its documentation were **generated with GLM 5.3 Flash** (Z.ai), used as the
implementation assistant throughout the project.

Key references the design builds on:

- Xie et al., *Are VLMs Ready for Autonomous Driving? An Empirical Study from the Reliability,
  Data, and Metric Perspectives* (DriveBench, ICCV 2025) — [arXiv:2501.04003](https://arxiv.org/abs/2501.04003)
- Hendrycks & Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and
  Perturbations* (ImageNet-C) — [arXiv:1903.12261](https://arxiv.org/abs/1903.12261)
- Michaelis et al., *Benchmarking Robustness in Object Detection: Autonomous Driving when Winter
  is Coming* (COCO-C) — [arXiv:1907.07484](https://arxiv.org/abs/1907.07484)
- Northcutt, Athalye & Mueller, *Pervasive Label Errors in Test Sets Destabilize Machine Learning
  Benchmarks* — [arXiv:2210.17059](https://arxiv.org/abs/2210.17059)
- [DriveLM](https://github.com/OpenDriveLab/DriveLM) (ECCV 2024) — dataset & GVQA schema
- [Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset) — camera_box parquet format
- ISO 21448 (SOTIF) & ISO 26262 — fault classes (permanent / intermittent / transient) and
  triggering-condition thinking behind the temporal pattern engine

## Limitations (v1)

- Waymo adapter is annotation-only (image extraction from tfrecords is a separate concern)
- Layer D/E families are spec-only
- Root-level `--root` paths are recorded in manifests (for provenance); scrub manifests before
  publishing offspring outputs if paths are sensitive

## License

MIT — see [LICENSE](LICENSE).
