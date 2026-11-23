# A CPU Inside an LLM: Paper2Code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/papers/weight-compiled-circuits/2611.26450-cpu-inside-an-llm/notebook.ipynb)

**Paper:** Tanmay Sah, *A CPU Inside an LLM: Zero-Gradient Compilation of Exact Circuits into Pretrained Weights*, arXiv:2611.26450v1 (19 Nov 2026). [arXiv abs](https://arxiv.org/abs/2611.26450) · [PDF](https://arxiv.org/pdf/2611.26450) · [alphaXiv](https://www.alphaxiv.org/abs/2611.26450)

**Official code:** none public as of 23 Nov 2026. The paper describes a "release codebase" (`demo_minimal.py`, `patches/*.safetensors`, `src/circuits/arithmetic_auto/`, `src/sidecar/sparse_sidecar_runtime.py`) but gives no URL. arXiv, Hugging Face Papers and the author's GitHub and Hugging Face accounts list no repository. No unofficial reimplementation was found either. This notebook is written from scratch from the PDF.

## What's here
| File | What it is |
|---|---|
| `notebook.ipynb` | A Colab notebook that teaches the paper with real code. It uses `Qwen/Qwen3.5-9B` in 4-bit by default. `MODEL_SIZE="4B"` selects `Qwen/Qwen3.5-4B`, the paper's own Qwen model, and `MODEL_SIZE="small"` selects `Qwen/Qwen3.5-0.8B` for CPU. Without a GPU it falls back to the small model on its own. |
| `notebook_smoke_executed.ipynb` | A full CPU run with Qwen3.5-0.8B in fp32, outputs included. |
| `build_notebook.py` | Regenerates `notebook.ipynb`. |
| `run_smoke.py` | The smoke test (nbclient plus environment-variable overrides). |

## Notebook contents
1. The idea in plain English: why RMSNorm's per-token λ, bias-free GLUs, token mixing and width all fight exact circuits
2. Primitives in pure PyTorch, each equation followed by its code:
   - single-layer impossibility
   - the Mellin-GLU identity and its Δ = 0.55 quadrature
   - the two-layer scale-invariant step
   - a 6-digit adder that stays exact under a random 256× λ in every layer (with a single-scale ablation)
   - Indefinite-RoPE self-sink attention (signed vs. positive kernels)
   - memoryless GatedDeltaNet shifts
3. Width extension on a real Qwen3.5: λ statistics, the Ky Fan projection bound, and d → d + K with γ = √(d/(d+K))
4. The main method: a zero-gradient graft of an exact, guarded `a+b=` circuit (K = 65 wires) into Qwen3.5, tested on:
   - all 100 sums
   - mid-sentence use
   - false triggers (max |Δz| = 0.0)
   - NL KL with and without norm compensation
5. Limitations, exercises, and the paper's numbers

**Smoke run (CPU, Qwen3.5-0.8B, fp32):**
- Grafted model: 100/100 exact. The base model also scores 100/100; single-digit sums are easy for it.
- Plain text: the circuit's logit change Δz is exactly 0.0.
- NL KL against the base model: ≈ 0 with norm compensation. With inert padding to 2d it rises to 1.68 without compensation and stays ≈ 0 with it.
- Toy 6-digit adder: 100% under random λ; the single-scale ablation scores 0%.

Every claim is labelled 📄 paper / 🧪 simplification / 💭 interpretation.
