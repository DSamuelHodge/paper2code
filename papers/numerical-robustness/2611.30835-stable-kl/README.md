# The KL Divergence Is Never Negative, Except in PyTorch: Paper2Code guided tour

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/papers/numerical-robustness/2611.30835-stable-kl/notebook.ipynb)

**Paper:** Zhen Yang, Wei Tang, Chu Chen, Rui Qian, Yixuan Chen, Fan Yang, Kangning Cui. arXiv:2611.30835 (cs.LG, 23 Nov 2026), [PDF](https://arxiv.org/pdf/2611.30835v1).

**Official code:** [github.com/xi-xiaoran/stablekl](https://github.com/xi-xiaoran/stablekl). The code is BSD-3-Clause and the data CC BY 4.0. The paper's abstract links it ("Code and data"), and its README cites the paper. Following the Paper2Code rule, nothing here is reimplemented. The notebook is a guided tour of the official code, pinned to commit `4486b36`.

## Where to start in the official repo
| Path | What it is |
|---|---|
| `README.md` | Install, the three entry points, cost table, limitations. |
| `stablekl/__init__.py` | The public API: `kl_divergence(p, q, mode="ours"\|"rewrite")` replaces `torch.distributions.kl_divergence`; `kl_div(...)` replaces `F.kl_div`; `logits_kl(teacher, student)` computes KL(softmax‖softmax) from logits. |
| `stablekl/closed_form.py` | Dispatch for the 18 same-family pairs (interface A). |
| `stablekl/_closed/<family>.py` | One module per family. Each holds the rewritten formula with analytic gradients (`<Family>KLC`). Read `exponential.py` first (φ(x) = x − log1p(x) and the atanh series, eq. 1 and 3), then `normal.py` (eq. 2). |
| `stablekl/logits.py` | The logits path (eq. 4): `LogitsKL`, `_phi` = h(w), `_round_once`. |
| `stablekl/interface_b.py`, `api_b.py` | The correctly rounded `F.kl_div`, with a Ziv-style interval guard and `UncertifiedRounding`. |
| `benchmark/`, `docs/REPRODUCE.md` | Benchmark generation and scoring. Reproduction has four levels: 0 tests and figures (minutes) · 1 scoring, which needs the 10.8 GB HF dataset `xi-xiaoran/stablekl-benchmark` · 2 Exp. 1–3 · 3 Exp. 4 (GPU). |
| `experiments/exp1_geometric_fit/geo_fit.py` | Exp. 5.1 (CPU). |
| `experiments/exp2_beta_design/beta_ig.py` | Exp. 5.2 (CPU). |
| `experiments/exp3_identical_distillation/distill_gpu.py`, `distill_grid.py --smoke` | Exp. 5.3 (the full grid needs a GPU). |
| `experiments/exp4a_quantized_eval/run_eval.py` and `exp4b`–`exp4d` | Exp. 5.4 with Qwen2.5-1.5B-Instruct and GPT-2 (GPU, plus Hugging Face checkpoints). |

```bash
git clone https://github.com/xi-xiaoran/stablekl && cd stablekl
pip install -e ".[test]" && pytest tests -m "not slow"     # 25 passed in ~2 s on CPU (our run)
```

## What the notebook does
1. Clones the official repo at commit `4486b36` and installs it.
2. Reproduces the paper's opening example on the same FP32 inputs: Exponential with λq = fl(1.0001). The official code returns 0; stablekl and mpmath give 5.0013×10⁻⁹.
3. Prints the official source next to each key equation, and checks each against mpmath:
   - eq. 1 and 3: `ExponentialKLC.forward`
   - eq. 2 and ∂KL/∂σp = e(e+2)/σp: `NormalKLC.forward`
   - eq. 4: `LogitsKL.forward` / `backward` and `_phi`
4. Draws a small version of Figure 1 (Exponential and Normal, FP16 and FP32): relative error of OFFICIAL, OFFICIAL+ and OURS against the true KL.
5. Separates the two error sources of interface B. Operator error is what `stablekl.kl_div` fixes. Input-representation error is what only `logits_kl` avoids.
6. Runs the official experiment code at small scale:
   - Exp. 5.1, Geometric fit (`geo_fit.run`)
   - Exp. 5.2, the Beta values
   - Exp. 5.3, identical-start distillation (`distill_gpu.run`): V = 32,768 on CPU, V = 151,936 on GPU
7. Exp. 5.4 Part A uses **the paper's model, Qwen2.5-1.5B-Instruct, in FP16** against an 8-bit RTN copy (the repo's `rtn()`), on WikiText-2. The reference is the repo's `exact_kl()`. Switches:
   - `SMALL=1` loads Qwen2.5-0.5B-Instruct
   - `LLM_MODEL=` loads any other model
   - `N_WINDOWS=32` uses the paper's 16,384 tokens
   - `SKIP_LLM=1` skips this section

   On CPU the model runs in FP16 too. A 9B model is not needed, because the paper's own model is small.
8. Ends with cost, limitations, exercises and the paper's headline numbers. Every claim is labelled 📄 paper/repo, 🧪 simplification or 💭 interpretation.

**Glue added by me:** none was needed in the repo. The notebook only adds plotting, an mpmath reference for Exponential, Normal and Beta, and a small Figure-1-style sampler. These are marked 🧪.

## Smoke run (CPU, PyTorch 2.15.0, no GPU)
The full notebook ran in about 5.5 minutes with 0 errors (`notebook_smoke_executed.ipynb`). Results:
- **Exponential example:** official 0.0; stablekl 5.0013260e-9; mpmath 5.0013261e-9.
- **Geometric fit (FP16, p⋆ = 0.001):** at the optimum the official code gives loss 0.0977 and gradient −100. The paper reports 0.098 and −100. The repair gives exact zeros.
- **Beta, n = 10³ / 10⁴ / 10⁵:** official FP32 gives −1.49e-3 / −5.62e-3 / +0.307. The paper reports −1.5e-3 / −5.6e-3 / +0.31. stablekl matches the true values.
- **Identical-start distillation (V = 32,768, FP16, Adam lr 10⁻², seed 0):**
  - official `F.kl_div`: 41.8% top-1 disagreement after 300 steps (the paper's median over 10 seeds is 41.0%)
  - `Categorical`: 44.9%
  - logits path: 0, with no weight changed
- **Qwen2.5-1.5B-Instruct vs RTN-8 (1,024 tokens):**
  - official `F.kl_div`: mean error −5.6%, with 21.9% of tokens negative (the paper's RTN column, on 16,384 tokens: −4.4% and 12.3%)
  - `reduction="mean"` reports 0
  - `Categorical` is ∞ on 96% of tokens
  - logits path: 0 negative tokens, and it matches the FP64 reference to 3e-15 before the final FP16 rounding

## Caveats
- The 10.8 GB benchmark dataset is not in the repo, so the paper's error-rate tables (Table 1, interface-B rounding) are quoted, not re-scored here.
- Model checkpoints (GPTQ/AWQ) are not redistributed. Only the RTN variant runs here.
- The full Exp. 3 grid and Exp. 4 need a GPU.
- Five certification tests are `xfail`, and `logits_kl` is not interval-certified (README).
- The repairs are unoptimised: `ours` costs 3.4–12× the official call on GPU, Beta about 320× (README). Our CPU timing for 10⁶ Normal pairs: 2.6 ms official, 208 ms rewrite, 424 ms ours.
- BF16 results are exploratory (paper Sec. 7).
