# Language Models Can Rewrite Their Own Weights: Paper2Code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/papers/self-improving-models/2611.21885-self-rewriting-weights/notebook.ipynb)

**Paper:** Fan et al., *Language Models Can Rewrite Their Own Weights*, arXiv:2611.21885 (Nov 2026). [arXiv abs](https://arxiv.org/abs/2611.21885) · [PDF](https://arxiv.org/pdf/2611.21885) · [project page](https://www.jiajunfan.com/projects/self-rewriting/)

**Official code:** none is linked in the paper or on the project page. There is an unofficial, independent reimplementation at [DukeCaesar/models-rewrite-own-weights](https://github.com/DukeCaesar/models-rewrite-own-weights). This notebook is written from scratch from the PDF.

## What's here
| File | What it is |
|---|---|
| `notebook.ipynb` | A Colab notebook that teaches the paper with real code. It uses `Qwen/Qwen3.5-9B` in 4-bit (the paper's own model family), and `MODEL_SIZE="small"` switches to `Qwen/Qwen3.5-0.8B` for CPU or the free tier. |
| `notebook_smoke_executed.ipynb` | A CPU smoke run with Qwen3.5-0.8B and tiny budgets, outputs included. It shows the code runs, not paper-level results. |
| `build_notebook.py` | Regenerates `notebook.ipynb`. |
| `run_smoke.py` | The smoke test (nbclient plus environment-variable overrides). |

## Notebook contents
1. The idea in plain English
2. Each key equation followed by the code that implements it: grading (Eq. 1), edits and the SaveEdit rules (Eq. 2, Alg. 4), the paired test (Eq. 3, Alg. 3), `W - αuuᵀW`, the damaged copy (Eq. 4), and Gated DeltaNet's `dt_bias`
3. The 27B model's edits replayed on Qwen3.5 (the layer-8 direction removal; `dt_bias` +0.5), plus damage and repair by outlier clipping
4. A minimal self-rewriting session in which the model edits its own weights (Alg. 1/2), with a reserved-problem check (protocol 2)
5. Limitations and differences from the paper, exercises, and the paper's numbers

**Runtime:** a GPU (L4 or A100 recommended; a T4 should fit). Every claim is labelled as 📄 from the paper, 🧪 a simplification, or 💭 an interpretation.
