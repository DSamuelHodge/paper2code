import os, nbformat as nbf
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))
COLAB = "https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/papers/weight-compiled-circuits/blog-2026-11-looped-wsp/notebook.ipynb"
SHA = "2752cd9f2d2e1e3e30e6686b3dbc23d8bcab875c"

md(rf"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

# Paper2Code guided tour: *Looped Weight-Space Programming: LLM-Written Transformers for Long Algorithmic Reasoning*

**Work:** Abdelrahman Elsayed\*, Ahmed Heakl\*, Abdelrahman Shaker, Fahad Khan, Salman Khan, Claude Opus 5.5 (MBZUAI; \* equal contribution). Research blog post, 23 Nov 2026: [mbzuai-oryx.github.io/LoopedWSP](https://mbzuai-oryx.github.io/LoopedWSP/). **There is no arXiv paper.** The authors cite it as a `@misc` blog post, and the arXiv search finds no matching entry. Everything tagged 📄 below quotes the blog post or the repository README.

**Official code:** [github.com/mbzuai-oryx/LoopedWSP](https://github.com/mbzuai-oryx/LoopedWSP) (MIT). The repo belongs to the authors' lab organisation and is linked from the post. It holds:
- the written weights of every transformer
- `weights/run.py`, a numpy runner for the ten main networks
- `weights/more_networks/runner/looped.py`, a runner with numpy and torch backends for the rest
- the prompts Claude received

There is no training code, because the method has no training. This tour **does not reimplement anything**. It clones the official repo (pinned to commit `{SHA[:7]}`), runs its code, and opens up the weights to show how they work.

**No GPU and no LLM needed.** The transformers have 6 to about 270K weights and run in numpy on CPU. The LLM in this work (Claude Opus 5.5) was the *author* of the weights, not part of the model. 🧪 We do not re-run the weight-writing sessions, which needed Claude Code; Section 7 shows the prompts.

**Labels:** 📄 stated in the blog or repo · 🧪 a simplification or choice made in this notebook · 💭 my interpretation.
""")

md(r"""
## Contents
0. Setup: clone the official repo
1. The idea in plain English
2. The one generic looped block (`run.py`: `layer`, `forward`, `loops`)
3. The smallest network, taken apart: parity in 6 numbers per token
4. The maze transformer: embedding layout, neighbour-reading heads, the distance wave mod 3, pointers, the path, and "extra loops don't undo the answer"
5. Toy experiments: run the official networks (Hanoi, Checker Jumping, chains, graph search, BIG-Bench examples, CLRS / SAT / PlanBench)
6. A real Maze-Hard test maze
7. How the weights were written: the prompts
8. Limitations, exercises, numbers from the blog
""")

md(r"""
## 0. Setup
""")
code(rf"""
import os, sys, subprocess, time, math, random, inspect, json
import numpy as np
import matplotlib.pyplot as plt
REPO_URL = "https://github.com/mbzuai-oryx/LoopedWSP.git"
PIN = os.environ.get("LWSP_COMMIT", "{SHA}")   # 🧪 pinned for reproducibility; set to "" for the latest main
REPO = os.path.abspath("LoopedWSP")
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO], check=True)
if PIN:
    subprocess.run(["git", "-C", REPO, "checkout", "-q", PIN], check=True)
print(subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %ad %an"], capture_output=True, text=True).stdout)
WEIGHTS = os.path.join(REPO, "weights")
sys.path.insert(0, WEIGHTS)
import run                      # the official runner: python run.py <name> <input>
np.set_printoptions(linewidth=160, precision=3, suppress=True)
print("networks:", sorted(f[:-5] for f in os.listdir(os.path.join(WEIGHTS, "networks")) if f.endswith(".json")))
""")

md(r"""
## 1. The idea in plain English

📄 The authors asked Claude Opus 5.5 to find shortest paths through hard 30×30 mazes (Maze-Hard, shortest paths longer than 110 steps). It solved **26%** of the test mazes it was given. They then asked it to **write a transformer** that solves the mazes instead. In one Claude Code session, it set all **19,936 weights by hand**, with no training, in 39 minutes, for about **$19**. That transformer solves **100%** of the test mazes, and also every maze they generated at 60×60 and 100×100.

**The trick is the loop.** A normal transformer has one block per step of computation, so a 200-step search would need 200 blocks. Here **one small block runs again and again**. Each pass does one step of a shortest-path search, so a bigger maze needs more passes, not more weights. 📄 "With it, one block can run thousands to millions of steps, so the number of weights does not grow with problem size." The authors call this **looped weight-space programming**.

Why it is interesting:
- 📄 **The LLM can write the step but cannot run the chain.** Claude cannot carry out a 100+ step search in text, but it can design one step, check it piece by piece, and leave the repetition to the loop.
- 📄 **Gradient descent did not find these weights.** The same architecture, reset to random weights and trained on the same 1,000 mazes, solved 0% in every run.
- 📄 **Every number has a meaning.** Claude decided what each of a cell's 35 numbers stands for before it set any weight, so we can read the state after any loop. We do exactly that in Section 4.
""")

# ---------------- 2 ----------------
md(r"""
## 2. The one generic looped block

📄 Every network runs the same loop (from the docstring of `weights/run.py`). Only the config (`networks/<name>.json`) and the weights (`<name>.npz`) differ:

$$h^{(0)}_i = \sum_{f} E_f[\text{value}_f(i)] \qquad\text{(one embedding table per token field)}$$

repeat $R$ times, for each layer of the **one shared** block:

$$h \mathrel{+}= \mathrm{concat}_j\Big(\mathrm{softmax}\big(\tfrac{Q_jK_j^\top}{\sqrt{d_k}} + \text{mask}\big)V_j\Big)W_o, \qquad h \mathrel{+}= \mathrm{ReLU}(hW_1+b_1)W_2 + b_2$$

then $\text{logits} = hW_{\rm out} + b_{\rm out}$ and greedy decoding. Note what is **missing**: there is no LayerNorm or RMSNorm. 💭 This makes exact hand-written arithmetic possible, and 📄 the authors had to *add* RMSNorm for gradient training to run stably (Section 4 of the blog). The loop count $R$ comes from a generic rule in the config:

$$R = \text{const} + \text{lin}\cdot\lceil n/\text{div}\rceil + \text{log2}\cdot\lceil\log_2 n\rceil .$$

Here is the official source of those three functions:
""")
code(r"""
for fn in (run.layer, run.forward, run.loops):
    print(inspect.getsource(fn))
""")
code(r"""
def n_weights(name):
    # dense parameter count (every entry of every matrix and bias) and the number of non-zeros, from the .npz
    z = np.load(os.path.join(WEIGHTS, "networks", name + ".npz"))
    dense = sum(int(np.prod(z[k])) for k in z.files if k.endswith(".shape"))
    nnz = sum(len(z[k]) for k in z.files if k.endswith(".val"))
    return dense, nnz
rows = []
for name in sorted(f[:-5] for f in os.listdir(os.path.join(WEIGHTS, "networks")) if f.endswith(".json")):
    cfg = json.load(open(os.path.join(WEIGHTS, "networks", name + ".json")))
    dense, nnz = n_weights(name)
    rows.append((name, cfg["d_model"], len(cfg["layers"]) if isinstance(cfg["layers"], list) else "?", cfg["mask"], cfg["readout"], cfg["loops"], dense, nnz))
print(f"{'network':20s} {'d':>5s} {'layers':>6s} {'mask':>7s} {'readout':>7s}  {'loop rule':50s} {'weights':>9s} {'nonzero':>8s}")
for r in rows:
    print(f"{r[0]:20s} {r[1]:5d} {r[2]:6} {r[3]:>7s} {r[4]:>7s}  {str(r[5]):50s} {r[6]:9,d} {r[7]:8,d}")
print("\n📄 The blog quotes 19,936 weights for the maze transformer and 3,686 for Tower of Hanoi.")
print("💭 The maze figure matches the dense count above, while the Hanoi figure matches the non-zero count; the blog")
print("   does not say which count it uses for each network, so compare both columns.")
""")

# ---------------- 3 ----------------
md(r"""
## 3. The smallest network, taken apart: parity

The `parity` network has $d = 6$ numbers per token and one layer with one attention head. 📄 Claude checked it only on strings of up to 20 bits, "yet it is correct at every length we tried, up to 4,095 bits". Trained models "fell to chance (45.5–53.8%) on 21 to 128 bits". Let us print **every** weight.
""")
code(r"""
def dense_mats(name):
    z = np.load(os.path.join(WEIGHTS, "networks", name + ".npz")); out = {}
    for k in sorted({f.rsplit(".", 1)[0] for f in z.files}):
        a = np.zeros(int(np.prod(z[k + ".shape"]))); a[z[k + ".idx"].astype(np.int64)] = z[k + ".val"]
        out[k] = a.reshape(tuple(int(s) for s in z[k + ".shape"]))
    return out
P = dense_mats("parity")
for k, v in P.items():
    print(f"--- {k} {v.shape}"); print(v[:6] if k == "emb.pos" else v)
""")
md(r"""
💭 **Reading the weights.** The six numbers of a token are $[1,\ b,\ \pi,\ s,\ p,\ p^2]$: a constant, the bit $b$ (`emb.tok`; BOS has $b = 0$), the running parity $\pi$, a scratch slot $s$, and the position $p$ and $p^2$ (`emb.pos`).

**Attention reads the previous token.** With $q = [141.42\,(p-1),\ -70.71]$ and $k = [p',\ p'^2]$,

$$\frac{q\cdot k}{\sqrt 2} = 100(p-1)p' - 50p'^2 = -50\big(p' - (p-1)\big)^2 + \text{const}(p),$$

so the score peaks at $p' = p-1$, with a gap of at least 50 to every other key. The head copies that token's parity $\pi$ into the scratch slot $s$ (`Wv`, `Wo`). This is the same "attention as an exact address lookup" that the maze heads use with $-100[\dots]$ (📄 Section 1 of the blog).

**The MLP computes XOR with ReLUs.** Neurons 0–3 compute $\mathrm{ReLU}(2(b+s) - c)$ for $c = 0.5, 1.5, 2.5, 3.5$, combined with signs $+,-,-,+$. That gives $1$ when $b+s = 1$ and $0$ when $b+s \in \{0, 2\}$, i.e. $b \oplus s$, written into $\pi$. Neurons 4–5 erase the old $\pi$ through $x - \mathrm{ReLU}(x) + \mathrm{ReLU}(-x) = 0$, and 6–7 erase the scratch $s$. 📄 The maze uses the same clean-up identity for its scratch numbers.

**The loop spreads the answer.** The mask is causal and $R = n - 1$. After loop $t$, token $p$ holds the parity of bits $1..p$ for every $p \le t+1$. The readout $\text{logit}_1 = 2\pi - 1$ reads the last token. Let us check this numerically, including on lengths far beyond the 20 bits Claude checked.
""")
code(r"""
net_par = run.load("parity")
def parity_trace(bits):
    toks = run.prompt("parity", net_par["cfg"]["io"], bits)
    T = np.asarray(toks); T = np.column_stack([T, np.arange(len(T))])
    h = net_par["emb"][0][T[:, 0]] + net_par["emb"][1][T[:, 1]]
    states = [h[:, 2].copy()]
    for t in range(run.loops(net_par["cfg"]["loops"], len(toks), len(toks))):
        for L in net_par["layers"]: h = run.layer(L, h, None)
        states.append(h[:, 2].copy())
    return np.array(states)            # [loops+1, tokens]: the running-parity slot π
bits = "1101001"
S = parity_trace(bits)
print("bits      :", " " + " ".join(bits))
for t, row in enumerate(S):
    print(f"after {t} loops:", " ".join(f"{v:.0f}" for v in row))
print("true prefix parities:", " ".join(str(sum(map(int, bits[:i])) % 2) for i in range(len(bits) + 1)))
""")
code(r"""
# Official generate() on random strings, far past the 20 bits Claude checked (🧪 lengths capped to keep CPU time small)
MAX_PARITY_LEN = int(os.environ.get("MAX_PARITY_LEN", "256"))
rng = random.Random(0)
for n in [8, 20, 21, 64, 128, 256, MAX_PARITY_LEN]:
    ok, t0 = 0, time.time()
    for _ in range(5):
        b = "".join(rng.choice("01") for _ in range(n))
        gen = run.generate(net_par, run.prompt("parity", net_par["cfg"]["io"], b))
        ok += run.answer("parity", net_par["cfg"]["io"], gen) == str(b.count("1") % 2)
    print(f"{n:5d} bits: {ok}/5 correct  ({time.time() - t0:.1f}s)")
""")

# ---------------- 4 ----------------
md(r"""
## 4. The maze transformer

📄 Each cell of the maze is one token, so a 30×30 maze is 900 tokens. Each token carries 35 numbers, and one block with **four attention heads** runs $R = 2n$ loops ($n$ = number of cells). The mask is `none` (bidirectional), and the readout scores every cell.

### 4.1 The embedding layout, read from the weights

📄 "They put $1$, $r_c$, $c_c$ and $r_c^2 + c_c^2$ into its first four numbers and its type into the next three." 💭 Reading `emb.*` and `out.W` gives the rest of the layout:

| dims | meaning |
|---|---|
| 0–3 | $1,\ r,\ c,\ r^2+c^2$ |
| 4, 5, 6 | wall, start, goal |
| 7, 8, 9 | $m_0, m_1, m_2$: one-hot of the distance to the goal **mod 3** (the goal starts with $m_0 = 1$) |
| 10–13 | pointers $p_\delta$ for down, left, right, up |
| 14 | `on`: the path flag (the start starts with 1) |
| 15–34 | 20 scratch numbers for what the heads copy |

💭 The 📄 readout equations $\ell_\# = \text{wall}$, $\ell_{\rm space} = 1-\text{wall}-\text{start}-\text{goal}$, $\ell_S = 3\,\text{start}$, $\ell_G = 3\,\text{goal}$, $\ell_o = 2\,\text{on}$ show up directly in `out.W`.
""")
code(r"""
net_mz = run.load("maze"); M = dense_mats("maze")
print("charset:", repr(net_mz["cfg"]["io"]["charset"]))
print("emb.char (rows = '#',' ','S','G','o'), first 15 dims:\n", M["emb.char"][:, :15])
print("emb.row[3][:4] =", M["emb.row"][3][:4], "  emb.col[5][:4] =", M["emb.col"][5][:4])
Wout = M["out.W"]; cs = net_mz["cfg"]["io"]["charset"]
print("readout:", {repr(cs[j]): {int(i): Wout[i, j] for i in np.nonzero(Wout[:, j])[0]} for j in range(len(cs))})
dense, nnz = n_weights("maze"); print(f"maze weights: {dense:,} (📄 19,936), of which non-zero: {nnz:,}")
""")
md(r"""
### 4.2 Four heads, each reading one neighbour

📄 The head that looks in direction $\delta = (\delta_r, \delta_c)$ scores every cell $n$ by

$$\frac{q_c\cdot k_n}{\sqrt 8} = -100\big[(r_n - r_c - \delta_r)^2 + (c_n - c_c - \delta_c)^2\big] + \text{const}(c),$$

"so the softmax puts a weight of $1 - O(e^{-100})$ on it". We check this from the official weights: for one cell, we compute each head's scores against every cell of a 10×10 grid and look at which offset wins and by how much.
""")
code(r"""
n_side = 10
toks = [[r, c, 1] for r in range(n_side) for c in range(n_side)]         # all open cells
h0 = sum(net_mz["emb"][f][np.array(toks)[:, f]] for f in range(3))
L0 = net_mz["layers"][0]
Q, K = h0 @ L0["Wq"], h0 @ L0["Wk"]
cell = 4 * n_side + 5                                                  # cell (4, 5)
for j in range(4):
    s = Q[cell, 8*j:8*j+8] @ K[:, 8*j:8*j+8].T / np.sqrt(8)
    best = int(np.argmax(s)); second = np.sort(s)[-2]
    off = (best // n_side - 4, best % n_side - 5)
    pred = -100 * (((np.arange(n_side*n_side) // n_side) - 4 - off[0])**2 + ((np.arange(n_side*n_side) % n_side) - 5 - off[1])**2)
    print(f"head {j}: reads offset {off}, score gap to runner-up = {s[best] - second:.1f}, "
          f"matches -100[...] + const: {np.allclose(s - s[best], pred - pred[best])}")
""")
md(r"""
### 4.3 The wave, mod 3, and the path, loop by loop

📄 With $\tau(z) = \mathrm{ReLU}(z) - \mathrm{ReLU}(z-1)$ (flat 0 for $z\le 0$, flat 1 for $z \ge 1$), two "wave neurons" per $k$ compute

$$m_{k+1}(c) \mathrel{+}= \tau\Big(2\!\!\sum_{n\in\mathcal N(c)}\!\! m_k(n) - 8\big[\mathrm{wall}(c)+m_0(c)+m_1(c)+m_2(c)\big] - \tfrac12\Big),$$

so after loop $t$ every cell within $t$ steps of the goal has a count. Because neighbouring open cells differ in distance by exactly 1,

$$d(n) = d(c) - 1 \iff d(n) \equiv d(c) - 1 \pmod 3,$$

so three binary numbers suffice for any maze size (mod 2 would fail). 📄 Pointers pick the first neighbour one step closer, in the order down, left, right, up. The path flag then grows from S: $\mathrm{on}(c) \mathrel{+}= \sum_\delta \tau(2\,\mathrm{on}(n_\delta) + 2\,p_{-\delta}(n_\delta) - 2\,\mathrm{on}(c) - 2.5)$. With $D = d(S)$, the wave reaches S at loop $D$ and the path is complete at loop $2D$.

Below we run the official layer function loop by loop on a random maze (`run.random_input`), decode dims 7–9 and 14 after each loop, and compare them with an independent BFS.
""")
code(r"""
MAZE_SIDE = int(os.environ.get("MAZE_SIDE", "15"))
# run.random_input draws a fresh random maze on every run (it uses an unseeded random.Random())
q = run.random_input("maze", MAZE_SIDE)
print(q)
toks = run.prompt("maze", net_mz["cfg"]["io"], q)
grid = q.split("\n"); n = len(grid); cells = n * n
# independent BFS from G (the oracle)
G = next((r, c) for r in range(n) for c in range(n) if grid[r][c] == "G"); S_ = next((r, c) for r in range(n) for c in range(n) if grid[r][c] == "S")
dist = {G: 0}; queue = [G]
for p in queue:
    for a, b in ((1, 0), (0, -1), (0, 1), (-1, 0)):
        x = (p[0] + a, p[1] + b)
        if 0 <= x[0] < n and 0 <= x[1] < n and grid[x[0]][x[1]] != "#" and x not in dist:
            dist[x] = dist[p] + 1; queue.append(x)
D = dist[S_]; R = run.loops(net_mz["cfg"]["loops"], len(toks), len(toks))
print(f"BFS: shortest path length D = {D};  loop rule R = 2n = {R}")

T = np.asarray(toks); h = sum(net_mz["emb"][f][T[:, f]] for f in range(3))
snap, mismatch, path_done, answers = {}, 0, None, []
for t in range(1, R + 1):
    h = run.layer(net_mz["layers"][0], h, None)
    m = h[:, 7:10]; on = h[:, 14]
    for i in range(cells):
        rc = divmod(i, n)
        if rc in dist and dist[rc] <= t:               # visited by loop t -> its count must equal d mod 3
            mismatch += int(np.argmax(m[i]) != dist[rc] % 3 or abs(m[i].sum() - 1) > 1e-9)
    answers.append(tuple(np.argmax(h @ net_mz["W_out"] + net_mz["b_out"], -1)))   # what it would write if it stopped now
    if t in (1, D // 2, D, (3 * D) // 2, 2 * D):
        snap[t] = (np.where(m.sum(1) > 0.5, m.argmax(1), -1).reshape(n, n), on.reshape(n, n).copy())
print(f"count == BFS distance mod 3 on every visited cell, every loop: {mismatch == 0}  (mismatches: {mismatch})")
solved_at = next(t + 1 for t in range(R) if all(a == answers[-1] for a in answers[t:]))
print(f"readout first equals the final answer (and never changes again) at loop {solved_at}   (📄 the path is complete at loop 2D = {2 * D})")
fig, axes = plt.subplots(1, len(snap), figsize=(3.1 * len(snap), 3.2))
for ax, (t, (cnt, on)) in zip(np.atleast_1d(axes), snap.items()):
    img = np.where(np.array([[ch == "#" for ch in row] for row in grid]), -2, cnt).astype(float)
    ax.imshow(img, cmap="viridis", vmin=-2, vmax=2); ax.contour(on, levels=[0.5], colors="orange")
    ax.set_title(f"loop {t}"); ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("distance mod 3 (colour; walls dark) and path flag (orange), read from the transformer's state"); plt.show()
""")
code(r"""
# The official answer and checker, then "extra loops don't undo the answer" (📄 Section 1, Figure 6)
gen = run.generate(net_mz, toks); ans = run.answer("maze", net_mz["cfg"]["io"], gen)
print(ans)
print("check (valid path, shortest, dataset tie-break):", run.maze_check(q.replace("\n", ""), ans.replace("\n", ""), n))
h_more = h.copy()
for _ in range(100): h_more = run.layer(net_mz["layers"][0], h_more, None)
print("after 100 extra loops, state dims 0-14 unchanged:", np.allclose(h_more[:, :15], h[:, :15]),
      "| max change:", np.abs(h_more[:, :15] - h[:, :15]).max())
""")

# ---------------- 5 ----------------
md(r"""
## 5. Toy experiments: run the official networks

These are the repo's own commands (📄 `weights/README.md`). Each one prints the answer and checks it with an independent simulator. 📄 "On a laptop, a 30×30 maze takes about 30 seconds and 10 Hanoi disks about a second."
""")
code(r"""
def official(*args, timeout=900):
    t0 = time.time()
    r = subprocess.run([sys.executable, "run.py", *args], cwd=WEIGHTS, capture_output=True, text=True, timeout=timeout)
    out = (r.stdout + r.stderr).strip().splitlines()
    shown = [l if len(l) < 300 else l[:300] + f" ... ({len(l):,} chars)" for l in out]
    print(f"$ python run.py {' '.join(a if len(a) < 60 else a[:57] + '...' for a in args)}   [{time.time() - t0:.1f}s]")
    print("\n".join(shown[-6:])); print()
HANOI_DISKS = os.environ.get("HANOI_DISKS", "10")
official("hanoi", HANOI_DISKS)                       # 📄 checked by Claude up to 6 disks; 20 disks = 1,048,575 moves
official("checker", "5")
official("chains", "x3 = 7; x5 = x3 + 4; x9 = x5 - 2; ? x9")
official("chains", "--random", "24")                # 📄 checked on chains up to 8 steps
official("graph", "--random", "30")
official("boolean_expressions", "( ( False ) ) or ( True ) is")
official("dyck_languages", "Complete the rest of the sequence, making sure that the parentheses are closed properly. Input: [ { [ ( ) ] }")
""")
md(r"""
📄 The `more_networks/` folder holds the rest: Sudoku-Extreme, GraphWalks, CLRS-Text, SATLIB, PlanBench, and the weight-capped mazes. It has its own runner with `numpy`, `numpy32`, `torch32` and `torch64` backends. "The CLRS-Text and PlanBench examples take seconds on a laptop. Sudoku and the mazes run thousands of loops and take minutes; on a GPU, add `--backend torch64`."
""")
code(r"""
MORE = os.path.join(WEIGHTS, "more_networks")
def official_more(folder, backend=None, timeout=1800):
    args = [sys.executable, "runner/looped.py", folder, f"{folder}/example_input.txt"] + (["--backend", backend] if backend else [])
    t0 = time.time(); r = subprocess.run(args, cwd=MORE, capture_output=True, text=True, timeout=timeout)
    print(f"$ python runner/looped.py {folder} {folder}/example_input.txt   [{time.time() - t0:.1f}s]")
    print("\n".join((r.stdout + r.stderr).strip().splitlines()[-4:])); print()
for folder in ["clrs_lcs_length", "clrs_scc", "clrs_mst_prim", "planbench_blocksworld"]:
    official_more(folder)
RUN_SLOW = os.environ.get("RUN_SLOW", "0") == "1"   # 🧪 off by default: minutes on CPU
if RUN_SLOW:
    import importlib.util
    backend = "torch64" if importlib.util.find_spec("torch") and __import__("torch").cuda.is_available() else None
    for folder in ["satlib", "maze_size_1k", "sudoku_extreme"]:
        official_more(folder, backend)
""")

# ---------------- 6 ----------------
md(r"""
## 6. A real Maze-Hard test maze

The test split of [`sapientinc/maze-30x30-hard-1k`](https://huggingface.co/datasets/sapientinc/maze-30x30-hard-1k) stores each maze as one 900-character string, which `run.py` accepts directly. 📄 The authors ran all 1,000 test mazes once and solved 100%, counting any shortest path as solved (the LDT paper's scoring). 🧪 We run only `N_TEST_MAZES` of them (each takes about a minute on CPU in float64 numpy), so this is a spot check, not a reproduction.
""")
code(r"""
import csv, urllib.request
N_TEST_MAZES = int(os.environ.get("N_TEST_MAZES", "2"))
url = "https://huggingface.co/datasets/sapientinc/maze-30x30-hard-1k/resolve/main/test.csv"
path = "maze_test.csv"
if not os.path.exists(path): urllib.request.urlretrieve(url, path)
rows = list(csv.DictReader(open(path)))
print(len(rows), "test mazes; columns:", list(rows[0].keys()))
for i in range(N_TEST_MAZES):
    qz = rows[i]["question"]; tk = run.prompt("maze", net_mz["cfg"]["io"], qz)
    t0 = time.time(); g = run.generate(net_mz, tk); a = run.answer("maze", net_mz["cfg"]["io"], g).replace("\n", "")
    valid, shortest, same = run.maze_check(qz, a, 30)
    print(f"test maze {i}: valid {valid}, shortest {shortest}, equals dataset answer {a == rows[i]['answer']}, "
          f"path length {a.count('o') + 1}  [{time.time() - t0:.0f}s, R = {run.loops(net_mz['cfg']['loops'], len(tk), len(tk))} loops]")
""")

# ---------------- 7 ----------------
md(r"""
## 7. How the weights were written: the prompts

📄 Each transformer was written by Claude Opus 5.5, in a Claude Code session, from a task description. The `prompts/` folder has these word for word. The rules are the heart of the method: **no gradient-based optimisation, every tensor set analytically by formulas, a generic loop and readout with no task logic in Python at inference, at most 20 DEV evaluation calls, and the TEST split run exactly once.** 📄 Table 1 of the blog: the maze build produced 196,771 output tokens, cost \$18.71 at list prices, and took 39 minutes and 3 revision rounds.

💭 You could try the same with a local open model, for example a Qwen ~9B. That is left as an exercise: the method needs a strong agentic coder that can run and test code over many rounds, and nothing in this repo depends on which LLM wrote the weights.
""")
code(r"""
brief = open(os.path.join(REPO, "prompts", "01_maze_hard_network_brief.txt")).read()
i = brief.find("## Hard rules"); print(brief[i:i + 2200])
""")

# ---------------- 8 ----------------
md(r"""
## 8. Limitations

- 🧪 **We run the released weights. We do not re-run the weight-writing**, which needed Claude Code sessions (18 sessions, 📄 \$1.19 to \$37.06 each).
- 🧪 **Spot checks, not reproductions.** We run a handful of inputs per task, and only `N_TEST_MAZES` real Maze-Hard mazes. The blog's 100% figures come from full test sets (e.g. 1,000 mazes; 141,040 Sudoku puzzles).
- 📄 **Each written transformer still has a size limit**, "set by its tables, its loop budget and its arithmetic precision rather than by the loop itself". For example, the maze position tables have 128 rows, so sides up to 128.
- 📄 **Compute.** The Sudoku transformer runs all 40,000 loops on every puzzle, "about 50 times more arithmetic per puzzle than TRM". 20-disk Hanoi takes about 9×10¹³ operations.
- 📄 **Known algorithms only.** "This readability depends on knowing the algorithm in advance", which "suits tasks with a known algorithm better than open-ended ones such as language."
- 📄 **Training caveat.** The gradient-descent runs used about 5 PFLOPs, so "these runs show that gradient descent did not find the weights under the settings tested, not that the weights cannot be learned."
- 💭 The layout table in 4.1 beyond dims 0–6 is my reading of the weights, confirmed by the BFS check in 4.3.

## Exercises
1. **Break the margin.** In the maze, scale `Wq` of one head by 0.05 so the score gap falls from 100 to 5. How many loops until the state goes wrong?
2. **Mod 2 instead of mod 3.** Explain from the equation in 4.3 why a mod-2 count cannot pick the closer neighbour, then find a maze where the tie-break would fail.
3. **Parity, written by you.** Write your own $d = 6$ parity weights from scratch. Use the official file format (`.npz` with `.shape`, `.idx`, `.val`) and test them with `run.generate`.
4. **Train it.** Reset `parity` to random weights and train it with PyTorch on strings of ≤ 20 bits, keeping $R = n - 1$. Does it generalise to 128 bits? (📄 the blog's trained models fell to chance.)
5. **Overthinking.** Run the chains network with 4× and 16× its loop count (edit the loop rule) and confirm the answer does not change.
6. **GPU backend.** On a GPU runtime, set `RUN_SLOW=1` and run Sudoku-Extreme with `--backend torch64`.

## Appendix: numbers from the blog (📄 quoted from the post or the repo README)

| claim | value |
|---|---|
| Maze-Hard, written transformer | 100% of test mazes; 19,936 weights; 39 min; about \$19 |
| Claude Opus 5.5 solving mazes in text | 26% (paper prompt); at most 50% with better prompts |
| Trained models on Maze-Hard | LDT 99.8% (90× the weights); EBR 94.1% (1.8M weights, 1,160 PFLOPs); TRM public checkpoint 91.6% |
| Gradient descent on the written architecture | 0% in every run (random init, same 1,000 mazes) |
| Smallest maze transformer under a weight cap | 928 weights, 100%, about 1,900× fewer than LDT and EBR |
| Bigger mazes, same weights | 100% on 50 mazes at 60×60 and 40 at 100×100; Claude checked up to 45×45 |
| Loops | 1,800 on 30×30 by default; 306 loops already solve every test maze |
| Tower of Hanoi | 3,686 weights; 20 disks = 1,048,575 moves, zero mistakes; over 2 million loops |
| Sudoku-Extreme | 141,040 of 141,040 test puzzles; Sotaku v2 98.96% on the first 10,000 |
| GraphWalks (256k–1M tokens) | 100% F1 on 200 problems; Claude Opus 5.5 66.8%; Gemini 4 Carbon 96.2% |
| BIG-Bench Hard / Extra Hard | 100% on Boolean Expressions, Dyck Languages, Web of Lies, Shuffled Objects |
| Parity / chains | correct up to 4,095 bits (checked ≤ 20); chains correct to 63 steps (checked ≤ 8) |
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.validate(nb)
nbf.write(nb, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook.ipynb"))
print("wrote", len(cells), "cells")
