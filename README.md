# Paper2Code

Runnable, teaching-first notebooks for machine-learning papers, sorted by topic. Each entry links the paper, says whether official code exists, and, when it does not, ships a Colab notebook that rebuilds the paper's core method with real, executable code. In every notebook, each claim is labelled 📄 *from the paper*, 🧪 *a simplification* or 💭 *an interpretation*, and numbers are quoted only from the PDF.

## The rule

1. **Check for official code first.** Look in the paper, the arXiv comments, the project page, Hugging Face Papers and the authors' GitHub.
2. **Official code exists:** the entry links it (status 🔗), and nothing is reimplemented. Optionally it adds a **guided tour** notebook (🧭) that clones the official repo, runs it, and explains the key equations next to the source that implements them.
3. **No official code:** build a from-scratch teaching notebook (status 📓). If an LLM is needed, it uses a real Qwen ~9B model in 4-bit, or the paper's own model when that is a Qwen of similar size. It always has a small-model switch and falls back to the small model when no GPU is present. It is smoke-tested on CPU before it is pushed. Unofficial reimplementations are linked but not relied on.
4. **No PDFs in the repo.** Link the arXiv page instead, or the project page for works without an arXiv entry.

## Index

### Self-improving models & model editing
`papers/self-improving-models/`

| Paper | arXiv / source | Date | Model | Colab / code | Status |
|---|---|---|---|---|---|
| [Language Models Can Rewrite Their Own Weights](papers/self-improving-models/2611.21885-self-rewriting-weights/) (Fan et al.) | [2611.21885](https://arxiv.org/abs/2611.21885) | 2026-11 | Qwen3.5-9B (4-bit); small: Qwen3.5-0.8B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/papers/self-improving-models/2611.21885-self-rewriting-weights/notebook.ipynb) | 📓 notebook · smoke-tested (CPU, Qwen3.5-0.8B) |

### Weight-compiled circuits & weight programming
`papers/weight-compiled-circuits/`

| Paper | arXiv / source | Date | Model | Colab / code | Status |
|---|---|---|---|---|---|
| [A CPU Inside an LLM: Zero-Gradient Compilation of Exact Circuits into Pretrained Weights](papers/weight-compiled-circuits/2611.26450-cpu-inside-an-llm/) (Sah) | [2611.26450](https://arxiv.org/abs/2611.26450) | 2026-11 | Qwen3.5-9B (4-bit); paper's model: Qwen3.5-4B; small: Qwen3.5-0.8B | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/papers/weight-compiled-circuits/2611.26450-cpu-inside-an-llm/notebook.ipynb) | 📓 notebook · smoke-tested (CPU, Qwen3.5-0.8B) |
| [Looped Weight-Space Programming: LLM-Written Transformers for Long Algorithmic Reasoning](papers/weight-compiled-circuits/blog-2026-11-looped-wsp/) (Elsayed, Heakl et al. (MBZUAI)) | [blog post](https://mbzuai-oryx.github.io/LoopedWSP/) | 2026-11 | no LLM needed (numpy looped transformers; weights written by Claude Opus 5.5) | [official code](https://github.com/mbzuai-oryx/LoopedWSP)<br>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/papers/weight-compiled-circuits/blog-2026-11-looped-wsp/notebook.ipynb) | 🔗 official code + 🧭 guided tour · smoke-tested (CPU, numpy) |

## Layout

```
papers/
  <category>/                     # kebab-case topic, created only when needed
    <arxiv-id>-<short-slug>/      # e.g. 2611.26450-cpu-inside-an-llm (no arXiv id: <venue>-<yyyy-mm>-<slug>)
      README.md                   # badge, paper links, code status, contents   (templates/paper_README.md)
      notebook.ipynb              # the Colab notebook (generated)
      build_notebook.py           # regenerates notebook.ipynb                  (templates/build_notebook_template.py)
      run_smoke.py                # CPU smoke test with the small model
      notebook_smoke_executed.ipynb
papers.yaml                       # the index: one entry per paper
scripts/build_readme.py           # regenerates this README from papers.yaml
templates/                        # templates for a new entry and for this README
```

## Contributing a paper

1. Check for official code (see the rule above). If it exists, add a `papers.yaml` entry with `official_code: <url>` and `status: official-code`. Either stop there, or add a guided-tour notebook that runs the official code. For a work without an arXiv entry, set `arxiv: false` and give `url` and `venue`.
2. Otherwise, pick or create a category, then copy `templates/paper_README.md` and `templates/build_notebook_template.py` into `papers/<category>/<arxiv-id>-<slug>/`.
3. Write the notebook in the template's order:
   - a plain-English overview
   - each key equation in LaTeX, followed by the code that implements it
   - a faithful minimal reimplementation of the main method
   - a toy experiment
   - limitations and exercises
   - an appendix of the paper's numbers, with table or figure references
4. Tag every claim 📄 / 🧪 / 💭. Never invent results.
5. Run `python build_notebook.py`, then `python run_smoke.py` on CPU with the small model. Commit both notebooks.
6. Add the entry to `papers.yaml` and run `python scripts/build_readme.py`. Check that the Colab badge opens.
