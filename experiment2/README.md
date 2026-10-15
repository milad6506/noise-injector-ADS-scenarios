# Experiment 2 — does parameter size gate the benefit of noise awareness?

**Question:** experiment 1 showed noise-awareness prompts make Qwen3-VL-8B substantially more
resilient. Does the same paragraph have the same effect on a **tiny** VLM — or is the benefit
itself a function of model capacity?

## Setup

| Element | Choice |
|---|---|
| **Tiny VLM (both arms)** | **SmolVLM-500M** (Q8_0 GGUF) served locally via `llama-server` with its mmproj vision projector — runs fully on-device, no API key |
| **Arms** | blind prompt vs noise-aware prompt — **identical except the awareness paragraph** (same prompt files as experiment 1) |
| **Data** | the same noise-injector batch (run `full10`), same harsh sampling (fog/glare/lens_dirt/snow/splash/low_light at S5) |
| **Judge** | GLM 5.3 Flash, blind-labeled, same protocol |
| **Protocol** | identical to experiment 1: mother-reference responses, deterministic deviation, per-cell judged verdicts |

## Running it with your own tiny VLM

`config.yaml` points the two arms at **any OpenAI-compatible endpoint** — local llama.cpp
(`llama-server -m <model>.gguf --mmproj <mmproj>.gguf --port 8081`), ollama, or a cloud endpoint.
Change `base_url`/`model`/`api_key_env` and rerun:

```bash
python3 experiment2/runner.py --run-dir runs/full10 --mothers 3 --per-mother 3 \
  --sample-strategy harsh --env-file <your-env>
```

Keys via env only; local servers need no key at all.

## Results (live run, 3 mothers × 3 severe children, 5:41 min)

| Model | Det. deviation blind → aware | Resilience blind → aware | Judge wins |
|---|---|---|---|
| **SmolVLM-500M** | 3.00 → 3.00 | 2.22 → 2.22 | tie 4 · aware 3 · blind 2 |
| **Qwen3-VL-8B** (exp 1, same cells) | 6.89 → **3.93** | 4.22 → **7.44** | aware 8 · tie 1 |

## Finding

**Parameter size gates the benefit of noise awareness.**

1. At 500M, the awareness paragraph had **zero behavioral effect** — both arms produced
   statistically indistinguishable responses (identical deviation/resilience/noise-handling means;
   near-chance judge verdicts). The tiny model simply does not act on the instruction.
2. The 500M model also **cannot reliably hold the output contract** (commands polluted with
   schema fields, sensor states missing) — its absolute resilience (2.22) sits far below the 8B's
   blind baseline (4.22). Awareness cannot help a model that cannot yet follow its task format.
3. At 8B, the same paragraph halves deviation and nearly doubles the resilience gap.

**Implication for the on-device model line:** noise awareness is not a free prompt trick — it is
a capability that emerges only once the model can (a) follow structured instructions and (b)
modulate its own perception reports accordingly. Tiny driving VLMs need the resilience trained-in
(the noise-injected fine-tuning route) rather than prompted-in.

*(full per-cell verdicts: `results/report.html` — with the cross-size comparison table)*
