# Experiment 1 — Can noise awareness make a driving VLM resilient?

**Question:** does simply *telling* the driving VLM that its sensor input line may carry
noise or faults change how it behaves when the noise injector corrupts the scenario — and
does a judge model measure that difference reliably?

## Setup

| Element | Choice | Why |
|---|---|---|
| **Target VLM (both arms)** | `qwen/qwen3-vl-8b-instruct` (small enough to be plausible on-device, good enough to drive) | user-specified |
| **Arm A — blind** | minimal prompt: *"You are the driver of the car…"* — outputs driving **commands**, **object list**, **per-camera sensor states** | baseline behavior |
| **Arm B — noise-aware** | **identical prompt + one paragraph**: the input line may contain noise/faults (fog, rain, freezes, dropped cameras, compression); treat degraded views as uncertainty; report camera health explicitly | the ONLY difference between the arms |
| **Judge** | `z-ai/glm-5.3-flash` (GLM) | blind to which arm is which |
| **Data** | a noise-injector run: mother scenarios (clean) + perturbed children across fault families | the injector generates everything |

## Protocol

1. Both arms see the **mother scenario** (clean reference) and the **child scenario** (perturbed).
2. Each arm answers for the child: `commands` (drive actions), `objects` (what it sees, per camera),
   `sensors` (per-camera health: ok / degraded / out).
3. **Deviation (deterministic)**: distance between an arm's *own* mother-response and its child-response
   (command Jaccard + object multiset similarity + sensor agreement). Expected deviation = reacting to a
   real injected fault; unexplained deviation = defect.
4. **Judge**: receives the mother ground truth + both arms' responses, blind-labeled, and scores
   `deviation / resilience / noise_handling` per arm, then picks a winner per child.
5. Aggregate: mean scores per arm + judge win counts.

## What we expect

- The blind arm should show **confident unexplained deviation** (hallucinated objects, commands changing
  without acknowledging faults) — the DriveBench failure mode.
- The noise-aware arm should **flag camera health** and keep commands closer to its mother behavior
  (resilience through acknowledgment, not through ignoring).

## Run it

```bash
# data first (the injector does this part)
noise-injector generate --dataset drivelm --data-root <your DriveLM root> \
  --samples 3 --family fog --severities 2 --out runs/exp1

# the experiment (keys via env only — never committed)
python3 experiment1/runner.py --run-dir runs/exp1 \
  --mothers 3 --per-mother 3 \
  --env-file <path-to-your-env>          # contains OPENROUTER_API_KEY=...

# outputs
experiment1/results/results.json
experiment1/results/report.html
```

## Results (live run, DriveLM × 3 mothers × {camera_dropout S2 flaky, frame_freeze S3 intermittent, jpeg S2 intermittent})

| Arm | Mean deviation (det.) | Mean resilience (judge) | Mean noise handling (judge) |
|---|---|---|---|
| qwen8b-blind | **6.97** | **4.22** | 3.56 |
| qwen8b-noise-aware | **3.84** | **6.22** | **5.11** |

**Judge (GLM 5.3 Flash) winners: noise-aware 8 / 9 children, blind 1 / 9.**

Reading: the ONLY difference between the arms is one paragraph of noise awareness — yet the
noise-aware arm deviates ~45 % less from its own clean-reference behavior, stays more resilient
under fault injection, and reports camera health far more often. The single blind-arm win came
from a child where both arms degraded similarly.

*(full per-cell verdicts: `results/report.html`)*

## Security

- API keys live **only** in environment variables (or a gitignored `experiment1/.env`)
- nothing in this repository contains, logs, or transmits a key anywhere except the model endpoint
- the judge model sees scenario data only — never credentials
