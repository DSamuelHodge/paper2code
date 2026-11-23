# Looped Weight-Space Programming: Paper2Code guided tour

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/papers/weight-compiled-circuits/blog-2026-11-looped-wsp/notebook.ipynb)

**Work:** Abdelrahman Elsayed\*, Ahmed Heakl\*, Abdelrahman Shaker, Fahad Khan, Salman Khan, Claude Opus 5.5 (MBZUAI; \* equal contribution), *Looped Weight-Space Programming: LLM-Written Transformers for Long Algorithmic Reasoning*. Research blog post, 23 Nov 2026: [mbzuai-oryx.github.io/LoopedWSP](https://mbzuai-oryx.github.io/LoopedWSP/). There is no arXiv paper; the authors cite it as a blog post (`@misc`).

**Official code:** [github.com/mbzuai-oryx/LoopedWSP](https://github.com/mbzuai-oryx/LoopedWSP) (MIT license). It is the authors' lab organisation and is linked from the post. Following the Paper2Code rule, nothing here is reimplemented. The notebook is a guided tour that runs the official code.

## Where to start in the official repo
| Path | What it is |
|---|---|
| `README.md` | Overview, results and how to run. |
| `weights/run.py` | One numpy script (about 370 lines) that runs the ten main transformers. `layer()`, `forward()` and `loops()` are the whole model: embedding → R × one shared block (softmax attention + ReLU MLP, no norm) → linear readout. Read this first. |
| `weights/networks/<name>.json/.npz` | Config (width, heads, mask, loop rule) and sparse weights: maze, hanoi, checker, parity, chains, graph, and four BIG-Bench tasks. |
| `weights/more_networks/` | Sudoku-Extreme, GraphWalks, CLRS-Text, SATLIB, PlanBench and the weight-capped mazes. They use `runner/looped.py`, which has numpy and torch (GPU) backends, and `runner/flops.py`. |
| `prompts/` | The task descriptions Claude Opus 5.5 received when it wrote each transformer, and the prompts for Claude solving mazes directly. |

To run it, you need only Python 3 and numpy:

```bash
git clone https://github.com/mbzuai-oryx/LoopedWSP && cd LoopedWSP/weights
python run.py maze --random 30     # ~1 min on CPU
python run.py hanoi 10             # < 1 s
```

**Compute needs:**
- Everything in `weights/` runs on a CPU.
- Sudoku and the large mazes take minutes; the README suggests `--backend torch64` on a GPU.
- There is no training code, because the method uses no training.
- The weights were written by Claude Code sessions, which are not reproducible from this repo.

## What the notebook does
1. Clones the official repo (pinned to commit `2752cd9`) and prints the official `layer` / `forward` / `loops` source next to the block equations.
2. Takes apart the 6-number-per-token **parity** network weight by weight: attention as an exact address lookup, XOR from ReLUs, and clean-up neurons. It checks parity strings far beyond the 20 bits Claude checked.
3. Walks through the **maze** transformer:
   - reads the 35-number cell layout from the embedding and readout weights
   - verifies the four neighbour heads' −100[…] score
   - runs the official layer loop by loop and checks the distance-mod-3 wave against an independent BFS
   - plots the wave and the path
   - confirms that extra loops change nothing
4. Runs the official CLI and runner on Hanoi, Checker Jumping, chains, graph search, BIG-Bench examples, CLRS-Text and PlanBench. SATLIB, Sudoku and the weight-capped mazes run with `RUN_SLOW=1`.
5. Spot-checks real **Maze-Hard** test mazes from Hugging Face (`N_TEST_MAZES`, default 2).
6. Shows the rules from the prompt Claude worked under, then limitations, exercises and the blog's numbers.

**No LLM and no GPU are needed.** The LLM in this work was the author of the weights, not part of the model. Every claim is labelled 📄 blog/repo, 🧪 simplification or 💭 interpretation.

**Smoke run:** the full notebook runs on CPU in about 2.5 minutes with no errors. It includes one real Maze-Hard test maze, which the transformer solved with the dataset's exact answer (path length 111, 1,800 loops, ~60 s).
