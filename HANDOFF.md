# HANDOFF — noise-injector project state

> Canonical state after the build-out sessions. Read this first when returning to the project.
> Repo: `milad6506/noise-injector-ADS-scenarios` (private; flip public whenever ready).
> Local work dir: `~/Documents/noise-injector-ADS-scenarios` (venv at `.venv`, Python 3.10).

## 1. What the tool is

`noise-injector` — deterministic fault-injection for ADS scenario data + agent harnesses,
plus an experiment framework for measuring VLM resilience.

**Layers implemented** (see `docs/TAXONOMY.md`, philosophy/algorithms in `docs/noise_types.html`):
- **A** sensor/image (21 pixel families + frame_freeze/camera_dropout/camera_desync), incl. the
  composite pattern `flaky_camera` (one seeded camera + Markov on/off)
- **B** annotation (misses, phantoms, jitter, class confusion, status flips, truncation + LLM description drift)
- **C** QA data (wrong answers, drift, category misassign, drop/dup, broken refs + LLM rewrite)
- **F** harness artifacts (prompt faults ×5, config faults ×7) — corrupts copies only, never executes models
- **D systemic + E adversarial** — spec-only (TAXONOMY §4–5), next rounds

**CLI**: `inject | harness (--link!) | batch (--preset!) | run (LLM-managed resilience rounds) |
generate (fetch→inject→report) | fetch (waymo via gsutil, carla public HF, drivelm gated w/ HF_TOKEN, lyft→link) |
report | preview | list-families`

**Key mechanics**: seed = sha256(scenario‖recipe) → byte-identical offspring; per-child manifests
record every injected event (severity now included) with paths redacted; reports are self-contained
interactive HTML: parent→children tree, bird's-eye 6-camera mosaic player with scrubber/play,
fog/rain visualized on canvas for annotation-only data, fault cells marked ⚠.

## 2. Experiment findings (all live runs, data committed under experimentN/results/)

| Experiment | Question | Result |
|---|---|---|
| **1** (exp1/) | Does noise-awareness prompting improve Qwen3-VL-8B under injected noise? | **Yes, severity-scaled**: harsh-structure 8/9 wins, dev 6.97→3.84; severe doses (S5) 7/9+1 tie, resilience 4.89→7.22 & 4.22→7.44 (re-run reproduced) |
| **2** (exp2/) | Same effect on a 500M model? | **No — capacity gates it**: SmolVLM-500M arms identical (dev 3.0 both, res 2.22, near-chance judge; output contract collapses). Implication: tiny VLMs need resilience *trained in*, not prompted |
| **3** (exp3/) | Does awareness of prompt corruption help? | **No — negative result**: safety 5.80 (simple) vs 5.08 (aware); benign Δ −0.68, adversarial Δ −0.90; followed-malicious **rises** 5/10→7/10 (re-run: 6/10); `section_delete` = 100% output collapse both branches |

Pattern: **sensor-line awareness scales with severity and capacity; prompt-integrity self-distrust
fails** — the latter needs structural defenses (signed/pinned instruction sections, output-contract
layering) — Layer-F round-2 work.

## 3. Environment facts

- OpenRouter key: local `~/Documents/DriveLM/judge_agent2/.env` — passed via `--env-file`, NEVER committed (verified repeatedly); env var name `OPENROUTER_API_KEY`
- Models: judge = `z-ai/glm-5.3-flash` (reasoning model! needs big max_tokens ~8000, content may be null → reasoning fallback), arms = `qwen/qwen3-vl-8b-instruct`; SmolVLM-500M served locally: `llama-server -m SmolVLM-500M-Q8_0.gguf --mmproj mmproj-Q8_0.gguf --port 8081` (dir `~/Documents/DriveLM/models_smolvlm`); ollama has `llama3.2:3b` (manager/engine) + `moondream` (vision target)
- OpenRouter 429s under load → all clients retry w/ backoff (exp runners + llm.py)
- `playwright` (+ chromium headless shell) installed in `.venv` for report verification; gsutil authed (Waymo GCS); HF DriveLM repo is gated (401 w/o `HF_TOKEN`); PDM_Lite CARLA packages are public
- Local DriveLM root: `~/Documents/DriveLM` (696 scenes; batch runs use `--data-root` there)

## 4. Gotchas (fixed — do not regress)

- Tests MUST NOT write into `experimentN/results/` — runners take `--results-file/--report-file`, tests use tmp_path (a pytest run once clobbered live results)
- Experiment runners import each other via **importlib with distinct module names** (`exp1_runner_protocol`) — plain `import runner` self-collides
- Report mosaic player: never clear the viewport if the mosaic exists (cached slots go detached); fault cells keyed by manifest `fault_cams`
- Large batches: use `batch` (splits per-family reports + index.html) — single-file reports get unwieldy past ~200 scenarios
- Manifest `params` hold curve values; the **severity int** is a separate event field (report chips fall back to folder-name parsing for old manifests)

## 5. Data sets on disk (local, gitignored)

- `runs/full10` — 10 DriveLM mothers → 980 children (complete_sensor preset, 913 s) + split reports
- `runs/drivelm-50` — 50 mothers × {flaky dropout, intermittent freeze, intermittent jpeg}
- `runs/waymo-100`, `runs/waymo-fog`, `runs/waymo-10` — Waymo camera_box batches
- `testset/*` — earlier demo testsets; `offspring/` demo from the first smoke tests

## 6. Ready next steps

1. Scale severe-dose exp1 (10 mothers × 6 harsh children ≈ 120 cells, few $) for statistical weight
2. Layer D (timestamped systemic faults — Waymo adapter is the carrier) + Layer E (poisoning/backdoors) implementation
3. YOLO target in the orchestrator (needs `ultralytics` install; `YoloTarget` class exists)
4. Noise-injected fine-tuning for SmolVLM (the exp2 conclusion motivates it: trained-in vs prompted-in)
5. CARLA closed-loop (FUTURE_DEVELOPMENT_PLAN.md §2)
6. Public release: flip repo visibility when the README reads as "benchmark result" — consider adding exp1–3 summary tables to the main README
