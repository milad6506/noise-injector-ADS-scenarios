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
| **B** | annotations (`key_object_infos`) | missed_detections, phantom_objects, bbox_jitter, class_confusion, status_flip, max_det_truncation |
| **C** | QA graph data | qa_wrong_answer, qa_answer_drift, qa_category_misassign, qa_drop, qa_duplicate, qa_crossref_break |
| **F** | agent-harness artifacts (prompts, config.json) — never executes models | prompt_truncation, prompt_section_delete, prompt_instruction_injection, prompt_garble, prompt_format_break, vocab_drop, vocab_misspell, vocab_swap, confidence_drift, max_tokens_starve, quality_drift, temperature_drift |

Layer D (systemic faults: latency, stuck-at, out-of-range, timestamp skew) and Layer E
(adversarial: poisoning, backdoors, targeted label flips) are specified in
[docs/TAXONOMY.md](docs/TAXONOMY.md) — implementation planned for the next round.

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
```

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

## Tests

```bash
pytest
```

The suite runs on synthetic fixtures — no dataset download required.

## Limitations (v1)

- Waymo adapter is annotation-only (image extraction from tfrecords is a separate concern)
- Layer D/E families are spec-only
- Root-level `--root` paths are recorded in manifests (for provenance); scrub manifests before
  publishing offspring outputs if paths are sensitive

## License

MIT — see [LICENSE](LICENSE).
