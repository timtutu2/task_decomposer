# Stage 1 Decomposer -- Regime A deployment guide

Goal: get `qwen3.6:27b` running locally via Ollama, then iterate through
v0 -> v1 -> v2 of the SayCan-style decomposer until the gold eval set
tells you whether Regime A is good enough, or whether you need to move
to fine-tuning (Regime B).

## 0. Hardware notes for your setup (2x RTX 5090 + 1x RTX 2080 Ti)

- Use **one** 5090 for this. 27B at Q4_K_M needs ~18GB -- comfortable
  headroom on a single 32GB 5090, no multi-GPU split needed.
- Leave the second 5090 free (it's not required here, and you'll likely
  want it free for Stage 2 / backbone work anyway).
- Don't bother pulling the 2080 Ti into this. Its memory/bandwidth would
  bottleneck a heterogeneous split for no benefit at this model size.
- If Ollama auto-spreads across both 5090s and you don't want that, pin
  it to one GPU:
  ```bash
  CUDA_VISIBLE_DEVICES=0 ollama serve
  ```

## 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version   # confirm >= 0.12.11 if you want logprobs later (v3)
```

## 2. Pull the model

```bash
ollama pull qwen3.6:27b
```

Default pull is Q4_K_M quantization. Since you have VRAM headroom on a
5090, it's worth checking the model's tag list on Ollama's library page
for a Q5_K_M or Q8_0 variant and pulling that instead -- for a
classification-style task (picking the exact right vocab entry), the
extra precision can matter more than it does for chat. Treat this as an
empirical question: run the eval harness (step 6) against both and keep
whichever scores higher on your gold set.

## 3. Smoke test

```bash
ollama run qwen3:32b "hello"
```

Confirm it's using the GPU you expect:

```bash
nvidia-smi   # while the above is running, check which GPU shows load
```

## 4. Python environment

```bash
cd decomposer
pip install -r requirements.txt
```

## 5. Files in this project

- `vocab.py` -- closed-vocabulary tables. **Currently placeholder values
  seeded only from the "open the box" example.** Replace `TARGET_PART`,
  `CONTACT`, `MOTION` with your HanDyVQA-derived tables before trusting
  results beyond this one smoke-test task.
- `schema.py` -- builds the JSON schema Ollama uses for constrained
  decoding (enum-restricted fields -- guarantees valid vocab output).
- `prompts.py` -- few-shot prompt construction for v0/v1/v2.
- `decomposer.py` -- the three modes: `run_v0`, `run_v1`, `run_v2`.
- `eval_harness.py` -- per-slot accuracy against a gold set.
- `example_tasks.json` -- starter gold set (one task). Expand this with
  more human-verified examples before the accuracy number means anything.
- `logprobs_probe.py` -- optional v3 step, soft-distribution scoring.

## 6. Run it

```bash
python decomposer.py
```

This runs all three modes against the one worked example and prints the
output of each, so you can eyeball whether the model is even getting the
right structure before you trust any numbers.

Then run the eval harness:

```bash
python eval_harness.py
```

This currently only has one gold task in `example_tasks.json`, so the
accuracy number isn't meaningful yet -- it's just confirming the
plumbing works end to end. Your next real step is building out a proper
gold set (the human-verified HanDyVQA + Goal-Step subset from your
dataset-construction plan) and dropping it into the same JSON format.

## 7. Decision point

Once you have a real gold set (dozens of tasks, not one):

- If v1/v2 per-slot accuracy is high enough for your purposes -- stop
  here. You don't need to fine-tune anything; Stage 1 just runs this
  pipeline once per episode as an offline preprocessing step.
- If accuracy is borderline, try: (a) more few-shot examples in the
  prompt, (b) v2 over v1 (sequential conditioning), (c) Q8 over Q4
  quantization.
- If accuracy is still insufficient after those, that's your signal to
  move to Regime B (LoRA-SFT on the silver-labeled dataset from the
  HanDyVQA-classifier + Goal-Step pipeline discussed earlier) -- don't
  build that pipeline speculatively before this number tells you it's
  needed.
