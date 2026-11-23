import nbformat as nbf
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/2611.21885-self-rewriting-weights/notebook.ipynb)

# Paper2Code: *Language Models Can Rewrite Their Own Weights*

**Paper:** Jiajun Fan, Neil He, Changyi Yang, Andy Mao, Xi Zhu, Chaoran Cheng, Kaiqu Liang, Shuaike Shen, Zetian Li, Ruihan Guo, Ge Liu. *Language Models Can Rewrite Their Own Weights.* arXiv:2611.21885v2, Nov 2026. [abs](https://arxiv.org/abs/2611.21885) · [pdf](https://arxiv.org/pdf/2611.21885) · [project page](https://www.jiajunfan.com/projects/self-rewriting/)

**Is there official code?** No. The paper and project page link no code repository. (The only GitHub link in the paper is reference [1], a third-party blog post on weight-space programming.) There is an **unofficial, independent** reimplementation at [DukeCaesar/models-rewrite-own-weights](https://github.com/DukeCaesar/models-rewrite-own-weights); by its own README, no numbers from the paper have been reproduced yet. This notebook is written from scratch from the PDF.

**What this notebook does.** It teaches the paper with code you can run. It uses **`Qwen/Qwen3.5-9B`**, the 9B member of the same **Qwen3.5** family the paper studies (2B / 4B / 9B / 27B), loaded in 4-bit so it fits one Colab GPU. Set `MODEL_SIZE = "small"` to swap in `Qwen/Qwen3.5-0.8B` for a quick run on CPU or the free tier.

**How to read the labels**
- 📄 **Paper:** stated in the PDF (numbers are quoted from it, with the table or figure).
- 🧪 **Simplification:** a choice this notebook makes to fit a Colab GPU; it is not the paper's setup.
- 💭 **Interpretation:** my reading of the paper where the text leaves details open.

> The paper finds that self-rewriting **emerges with scale**: only the 27B model improved or repaired itself, and the **9B model "breaks itself"** in the paper's sessions (📄 Fig. 2, Tab. 2). So with a 9B (or 0.8B) model, expect the *autonomous* session in Part 4 to fail most of the time. That failure matches the paper. To show what a *successful* edit looks like, Part 3 replays the edits the paper says the 27B model wrote (📄 Tab. 1, Tab. 13) and tests them on our model.
""")

md(r"""
## Contents
0. Setup and configuration
1. The idea in plain English
2. Key concepts, each as an equation followed by code
   - 2.1 Accuracy and grading, Eq. (1)
   - 2.2 Edits $\theta \oplus \Delta$ and the self-rewriting objective, Eq. (2), plus the SaveEdit rules (Alg. 4)
   - 2.3 The paired test, Eq. (3) and Alg. 3
   - 2.4 `remove_direction`: $W - \alpha uu^\top W$
   - 2.5 A damaged copy, Eq. (4), and finding the damage
   - 2.6 Gated DeltaNet's time-step bias (the session-8 edit)
3. Load the model and data, then replay the paper's edits on a real Qwen3.5
   - 3.1 Starting diagnosis
   - 3.2 Read the hidden states: finished vs. cut answers
   - 3.3 Edit A: remove the "unfinished" direction at an early layer (the 27B model's first-protocol edit)
   - 3.4 Edit B: time-step bias +0.5 in every linear-attention layer (session 8)
   - 3.5 Toy experiment: damage the model, then repair it with the 27B model's rule
4. Main method: a minimal self-rewriting session (Alg. 1 and 2), where the model edits **itself**
5. Limitations: how this toy differs from the paper
6. Exercises
7. Appendix: numbers from the paper
""")

md(r"""
## 0. Setup

**Runtime:** use a GPU runtime (Runtime → Change runtime type). The 9B model in 4-bit needs about 9–10 GB of GPU memory, because the output matrices stay in 16-bit (see below). An **L4 or A100** is the comfortable choice. A **T4 (16 GB)** should fit, but it has no bf16, so the notebook falls back to fp16 and runs slowly. Without a GPU, the notebook switches to the small model on its own.
""")

code(r"""
# Install dependencies (skipped if SKIP_INSTALL=1, e.g. when validated offline)
import os, sys, subprocess
if os.environ.get("SKIP_INSTALL") != "1":
    pkgs = ["transformers>=5.24", "accelerate", "datasets", "math-verify"]
    try:
        import torch
        if torch.cuda.is_available():
            pkgs.append("bitsandbytes>=0.45")
    except ImportError:
        pkgs.append("torch")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs], check=True)
    # Optional speed-up for Gated DeltaNet layers on GPU (correct but slower torch fallback is used otherwise):
    # subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flash-linear-attention", "causal-conv1d"])
print("ok")
""")

code(r"""
# ====== CONFIGURATION (edit here) ======
import os, torch

MODEL_SIZE = os.environ.get("MODEL_SIZE", "9B")   # "9B"  -> Qwen/Qwen3.5-9B in 4-bit (GPU)
                                                   # "small" -> Qwen/Qwen3.5-0.8B (CPU / free tier)
MODEL_IDS = {"9B": "Qwen/Qwen3.5-9B", "small": "Qwen/Qwen3.5-0.8B"}

HAS_GPU = torch.cuda.is_available()
if MODEL_SIZE == "9B" and not HAS_GPU:
    print("No GPU found -> switching to the small model.")
    MODEL_SIZE = "small"
MODEL_ID = MODEL_IDS[MODEL_SIZE]
LOAD_4BIT = (MODEL_SIZE == "9B") and HAS_GPU

# 🧪 Simplification: the paper uses a 16,384-token budget (4,096 in some sessions), 256-problem diagnoses,
# 20 forward passes and 200 code runs per session. We shrink everything so a toy run takes minutes, not hours.
E = lambda k, d: type(d)(os.environ.get(k, d))
MAX_NEW_TOKENS = E("MAX_NEW_TOKENS", 1024 if MODEL_SIZE == "9B" else 384)  # answer budget L
N_DIAG         = E("N_DIAG", 24 if MODEL_SIZE == "9B" else 8)     # starting-diagnosis problems N0
N_PAIRED       = E("N_PAIRED", 24 if MODEL_SIZE == "9B" else 8)   # problems per paired test
N_HELDOUT      = E("N_HELDOUT", 48 if MODEL_SIZE == "9B" else 8)  # held-out problems
BATCH_SIZE     = E("BATCH_SIZE", 12 if MODEL_SIZE == "9B" else 4)
READ_MAX_TOKENS= E("READ_MAX_TOKENS", 1024)
RUN_SESSION    = E("RUN_SESSION", 1)        # Part 4: let the model edit itself
SESSION_PASSES = E("SESSION_PASSES", 4)     # B (paper: 20)
SESSION_CODE_RUNS = E("SESSION_CODE_RUNS", 30)  # C (paper: 200)
SESSION_TURNS  = E("SESSION_TURNS", 16)     # Tmax (paper: 160)
EDITOR_MAX_NEW_TOKENS = E("EDITOR_MAX_NEW_TOKENS", 2048 if MODEL_SIZE == "9B" else 512)  # paper: 12,000

if HAS_GPU:
    DEVICE = "cuda"
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DEVICE = "cpu"; DTYPE = torch.bfloat16
torch.set_grad_enabled(False)   # 📄 no gradients anywhere: the model is its own optimizer
print(dict(MODEL_ID=MODEL_ID, LOAD_4BIT=LOAD_4BIT, DEVICE=DEVICE, DTYPE=str(DTYPE), MAX_NEW_TOKENS=MAX_NEW_TOKENS))
""")

md(r"""
## 1. The idea in plain English

A language model is normally **trained once and frozen**. To improve it you can retrain it, which takes gradients, an optimizer and a training pipeline. Or you can prompt it or spend more compute at inference, but that gain disappears once the task ends.

The paper shows a **third way, self-rewriting**. 📄 The model receives **its own weights**, **its graded answers on practice problems** and **the ability to run itself**. It then reasons, in a multi-turn session, about *why* it fails and writes Python that changes the numbers in its own weight tensors. **No gradient is computed and no optimizer runs**: every changed value comes from the model's own reasoning. Only forward passes are used, just as in serving.

📄 What the paper found with Qwen3.5-27B:
- **It surpasses itself.** Its own edit makes it **6.6 points** more accurate (57.2% → 63.8%) on **490 held-out problems** with eight answers each (session 8, second protocol; Fig. 1, Tab. 13). The edit raises the *time-step bias* of every linear-attention layer by 0.5, so fewer answers run out of tokens (33.6% → 26.1% cut).
- **It repairs a damaged copy of itself.** When 8 rows of one MLP output matrix are overwritten with huge noise, the copy scores 0.0%. The model locates the matrix by comparing largest and median weights, clips it, and brings the copy to **33.0%** on 200 held-out problems (base: 58.5%; the named matrix alone gives 55.0%) (Tab. 1, Fig. 15). The repair also carries over to images (7.3% → 70.2% on six image benchmarks, vs. 84.3% for the base model).
- **The ability emerges with scale.** 2B makes no edit; **4B and 9B add random noise and break themselves**; only 27B measures, tests and improves (Fig. 2, Tab. 2).

**The loop** (📄 Fig. 3, Alg. 1): diagnosis → model writes one action per turn (**code** over its weights, **generate**, a paired before/after test on practice problems, or **read**, its hidden states on texts it chooses) → keep the edit only if its tests support it → submit or withdraw → evaluate on held-out problems.
""")

md(r"""
## 2. Key concepts

### 2.1 Accuracy and grading, Eq. (1)

📄 With answer settings $\pi$ (prompt, sampling, budget of $L$ new tokens), an answer $y$ to problem $x$ is graded

$$g(x,y)=\begin{cases}1 & \text{if its thinking ends within } L \text{ tokens and its final answer matches the reference}\\ 0 & \text{otherwise}\end{cases}$$

$$\mathrm{Acc}(\theta;\mathcal D)=\mathbb E_{x\sim\mathcal D}\,\mathbb E_{y\sim M_\theta(\cdot\mid x;\pi)}\,g(x,y). \tag{1}$$

📄 Details from Tab. 5: the last `\boxed{}` *after* the thinking is checked with Math-Verify. An answer still thinking when the budget runs out earns nothing. A **looping** answer is one cut at the limit whose last 6,000 characters compress (zlib) to under 15% of their size. The paper's main failure mode is *answers that never finish*, not wrong reasoning: the 27B model's diagnosis showed "80.5% of answers hit the token limit … but 96% of completed answers are correct" (Fig. 13).
""")

code(r"""
import re, zlib

def split_thinking(text: str):
    # Qwen3.5 with thinking on: the prompt ends in '<think>\n', so the generation is '... </think> final answer'.
    if "</think>" in text:
        think, ans = text.split("</think>", 1)
        return think, ans, True
    return text, "", False  # thinking never closed

def last_boxed(s: str):
    # Content of the last \boxed{...}, with nested braces handled.
    i = s.rfind("\\boxed")
    if i < 0: return None
    j = s.find("{", i)
    if j < 0: return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{": depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0: return s[j+1:k]
    return None

def _norm(a):
    a = a.strip().strip("$").replace(" ", "").replace("\\!", "").replace("\\left", "").replace("\\right", "")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = a.replace("dfrac", "frac").replace("tfrac", "frac").rstrip(".")
    return a

try:
    from math_verify import parse, verify
    def answers_match(pred, gold):
        try:
            return bool(verify(parse("$" + gold + "$"), parse("$" + pred + "$")))
        except Exception:
            return _norm(pred) == _norm(gold)
except ImportError:
    def answers_match(pred, gold): return _norm(pred) == _norm(gold)

def is_loop(text: str, cut: bool) -> bool:
    # 📄 Tab. 5: cut at the limit and the last 6,000 chars compress to < 15% of their size.
    if not cut: return False
    tail = text[-6000:].encode()
    return len(tail) > 0 and len(zlib.compress(tail)) < 0.15 * len(tail)

def grade(text: str, gold: str) -> int:
    # g(x, y) of Eq. (1)
    _, ans, closed = split_thinking(text)
    if not closed: return 0
    b = last_boxed(ans)
    return int(b is not None and answers_match(b, gold))

# --- unit tests on strings (no model needed) ---
assert grade("Let me compute 6*7... </think> The answer is \\boxed{42}.", "42") == 1
assert grade("Let me compute ... \\boxed{42} and I keep thinking", "42") == 0          # thinking never closed
assert grade("ok </think> \\boxed{\\frac{1}{2}}", "\\dfrac{1}{2}") == 1
assert grade("ok </think> \\boxed{41}", "42") == 0
assert is_loop("I will write the response. " * 400, cut=True) and not is_loop("abc", cut=True)
print("grading helpers OK")
""")

md(r"""
### 2.2 Edits and the self-rewriting objective, Eq. (2)

📄 A model has weights $\theta=(W_k)_{k\in K}$, one tensor per name. An **edit** $\Delta$ replaces some tensors with tensors of the same shape, which gives $\theta\oplus\Delta$. Given practice problems $P\sim\mathcal D$, self-rewriting asks **the model itself** to write $\hat\Delta$ such that

$$\mathrm{Acc}(\theta\oplus\hat\Delta;\mathcal D_{\text{test}}) > \mathrm{Acc}(\theta;\mathcal D_{\text{test}}), \tag{2}$$

with $\mathcal D_{\text{test}}$ never seen while the edit is made, **no gradient with respect to $\theta$, and no optimizer**.

📄 **SaveEdit rules (Alg. 4):** refuse edits to the final norm, the input embedding and the output layer; refuse a new tensor that changes shape, has a non-finite entry, or **equals $cW_k$ for a constant $c$** (rescaling a whole tensor acts like changing the sampling temperature). Edits are stored in bf16.

🧪 Our edits live in a Python `dict` (name → tensor). They are swapped into the model only during the forward passes that request them, then swapped out again. This mirrors 📄 "its own weights never change, and the current edit acts only on the runs it requests" (Sec. 2.2).
""")

code(r"""
import torch
from contextlib import contextmanager

FORBIDDEN_SUBSTRINGS = ("embed_tokens", "lm_head", "model.norm.", "visual", "mtp")

def check_edit(name, W_old, W_new):
    # SaveEdit checks of Alg. 4; returns None if OK, else the reason for refusing.
    if any(s in name + "." for s in FORBIDDEN_SUBSTRINGS) or name in ("model.norm.weight",):
        return f"refused: {name} is the embedding, final norm or output layer"
    if tuple(W_new.shape) != tuple(W_old.shape):
        return f"refused: shape {tuple(W_new.shape)} != {tuple(W_old.shape)}"
    if not torch.isfinite(W_new).all():
        return "refused: non-finite values"
    a, b = W_old.double().flatten(), W_new.double().flatten()
    c = (a @ b) / (a @ a).clamp_min(1e-30)
    if torch.allclose(b, c * a, rtol=0, atol=1e-6 * float(a.abs().max()) + 1e-12):
        return f"refused: new tensor is {float(c):.4g} x the old one (whole-tensor rescale / no-op)"
    return None

@contextmanager
def use_weights(model, *edits):
    # Temporarily load theta ⊕ Δ1 ⊕ Δ2 ... (later edits override earlier ones), restore afterwards.
    merged = {}
    for e in edits:
        if e: merged.update(e)
    params = dict(model.named_parameters())
    saved = {}
    try:
        for k, v in merged.items():
            p = params[k]
            saved[k] = p.detach().to("cpu", copy=True)
            p.data.copy_(v.to(p.device, p.dtype))
        yield
    finally:
        for k, v in saved.items():
            params[k].data.copy_(v.to(params[k].device))

# --- tiny demo on a toy module ---
toy = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
W0 = toy[0].weight.detach().clone()
print(check_edit("0.weight", W0, 2 * W0))                         # refused (rescale)
print(check_edit("0.weight", W0, W0[:2]))                         # refused (shape)
print(check_edit("model.embed_tokens.weight", W0, W0 + 1))        # refused (embedding)
ok_edit = {"0.weight": W0 + 0.01 * torch.randn_like(W0)}
print(check_edit("0.weight", W0, ok_edit["0.weight"]))            # None = accepted
with use_weights(toy, ok_edit):
    assert not torch.equal(toy[0].weight, W0)
assert torch.equal(toy[0].weight, W0)
print("edit swap-in / swap-out OK")
""")

md(r"""
### 2.3 The paired test, Eq. (3)

📄 When a pass runs **both** weights, the unedited and edited model answer the **same problems with the same seeds**, so their answers form pairs. For $m$ problems with $n$ answers each and grades $g^0_{ij}, g^1_{ij}$ (without/with the edit):

$$d_i=\frac1n\sum_{j=1}^n\left(g^1_{ij}-g^0_{ij}\right),\qquad \hat\delta=\frac1m\sum_{i=1}^m d_i,\qquad \mathrm{SE}(\hat\delta)=\sqrt{\frac{1}{m(m-1)}\sum_{i=1}^m\left(d_i-\hat\delta\right)^2}. \tag{3}$$

📄 The test also reports the change in the share of answers that run out of tokens, plus the counts $n_\uparrow$ (wrong → right) and $n_\downarrow$ (right → wrong). Seeds are a hash of the problem identifier (Alg. 3). 📄 For held-out comparisons the paper uses a one-sided **sign test** over discordant pairs (App. D.3), and on its own task a sign-flip permutation test.

🧪 We generate in batches for speed, so the seed is a hash of the *batch's* problem ids (identical for both weights), not one seed per problem. The pairing (same problems, same seed, both weights) still holds.
""")

code(r"""
import hashlib, math, random

def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % (2**31)

def paired_stats(g0, g1):
    # g0, g1: lists (one per problem) of lists of 0/1 grades (n answers each). Implements Eq. (3).
    m = len(g0)
    d = [sum(b - a for a, b in zip(r0, r1)) / len(r0) for r0, r1 in zip(g0, g1)]
    delta = sum(d) / m
    se = math.sqrt(sum((x - delta) ** 2 for x in d) / (m * (m - 1))) if m > 1 else float("nan")
    n_up = sum(a == 0 and b == 1 for r0, r1 in zip(g0, g1) for a, b in zip(r0, r1))
    n_down = sum(a == 1 and b == 0 for r0, r1 in zip(g0, g1) for a, b in zip(r0, r1))
    return dict(delta=delta, se=se, n_up=n_up, n_down=n_down)

def sign_test_one_sided(n_up, n_down):
    # P(X >= n_up) for X ~ Binomial(n_up + n_down, 1/2): "is the edit better?"
    n = n_up + n_down
    if n == 0: return 1.0
    return sum(math.comb(n, k) for k in range(n_up, n + 1)) / 2**n

# --- synthetic check: an edit that flips 6 wrong->right and 2 right->wrong over 40 problems ---
rng = random.Random(0)
g0 = [[rng.random() < 0.5] for _ in range(40)]
g1 = [list(r) for r in g0]
flips_up = [i for i in range(40) if not g0[i][0]][:6]; flips_down = [i for i in range(40) if g0[i][0]][:2]
for i in flips_up: g1[i][0] = True
for i in flips_down: g1[i][0] = False
g0 = [[int(x) for x in r] for r in g0]; g1 = [[int(x) for x in r] for r in g1]
s = paired_stats(g0, g1); print(s, "sign-test p =", round(sign_test_one_sided(s["n_up"], s["n_down"]), 4))
assert abs(s["delta"] - 0.1) < 1e-9 and s["n_up"] == 6 and s["n_down"] == 2
""")

md(r"""
### 2.4 `remove_direction`: $W-\alpha\,uu^\top W$

📄 The session gives the model a helper that computes $W-\alpha uu^\top W$, "which weakens by the fraction $\alpha$ what a matrix $W$ writes along a unit direction $u$ of the hidden state" (Sec. 2.2). Each layer writes into the hidden state through **two output matrices**: the attention block's (`out_proj` for linear attention, `o_proj` for full attention) and the MLP's (`down_proj`).

Why it works: for any input $x$, the layer's contribution changes from $Wx$ to $(I-\alpha uu^\top)Wx$. Its component along $u$ is scaled by $(1-\alpha)$ and every orthogonal component is unchanged. The change has **rank one**.

📄 The 27B model's first-protocol edit (Tab. 1): $u$ is the unit direction from the mean layer-8 hidden state of 20 *finished* answers to that of 20 answers that *ran out of tokens*. It removed $\alpha=5\%$ of it from both output matrices of layer 8, "a change of 0.6% and 0.1% of their norm". Result: own task 22.4% → 29.6%; held-out 57.2% → 58.3% (both significant).
""")

code(r"""
def unit(v):
    v = v.float(); return v / v.norm().clamp_min(1e-12)

def mean_diff_direction(A, B):
    # unit vector from mean(B) to mean(A); A, B: [n, d]
    return unit(A.float().mean(0) - B.float().mean(0))

def remove_direction(W, u, alpha):
    # W: [d_hidden, d_in] (nn.Linear weight, writes into the hidden state); u: [d_hidden]
    u = unit(u).to(W.device)
    Wf = W.float()
    return (Wf - alpha * torch.outer(u, u @ Wf)).to(W.dtype)

# --- check the algebra on random numbers ---
torch.manual_seed(0)
d, k = 64, 128
W = torch.randn(d, k) / k**0.5; u = unit(torch.randn(d)); x = torch.randn(k)
W2 = remove_direction(W, u, 0.05)
print("component along u before/after:", float(u @ (W @ x)), float(u @ (W2 @ x)))   # scaled by 0.95
v = unit(torch.randn(d)); v = unit(v - (v @ u) * u)                                 # orthogonal to u
print("orthogonal component before/after:", float(v @ (W @ x)), float(v @ (W2 @ x)))
print("rank of the change:", torch.linalg.matrix_rank(W2 - W).item(),
      "| relative norm change: %.4f" % float((W2 - W).norm() / W.norm()))
""")

md(r"""
### 2.5 A damaged copy, Eq. (4), and finding the damage

📄 With $W$ the MLP output matrix of one layer and $\sigma$ the standard deviation of its entries, the damaged copy $\tilde\theta$ equals $\theta$ except on a block $R$ of consecutive rows:

$$\tilde W_{r,:}\sim\mathcal N\!\left(0,(\lambda\sigma)^2 I\right)\quad\text{for } r\in R,\qquad |R|=8,\ \lambda=200. \tag{4}$$

📄 In the 27B model this was `mlp.down_proj` of layer 30 of 64, rows 3068–3075. The model was told only that "some values were corrupted". Its diagnosis compared **the largest weight with the median weight** of every MLP and linear-attention output matrix. The damaged matrix stood at **1397×** its median; no other matrix exceeded 162.3×. It then **clipped every weight beyond 15× the median magnitude**. It applied that clip to the damaged matrix and to 19 output matrices of layers 0–10 (0.013% of their weights), which took the copy from 0.0% to 33.0%. The clip on the named matrix alone gave 55.0% (Tab. 1, Fig. 15). 📄 The repair does **not** restore the original values: in the damaged rows it caps the random values (largest |w|: 0.071 original, 10.1 damaged, 0.108 after repair).

Note that $W_{r,:}$ is **row** $r$ of the `nn.Linear` weight, i.e. the weights that produce output coordinate $r$ of the MLP.
""")

code(r"""
def damage_matrix(W, rows, lam=200.0, seed=0):
    # Eq. (4): overwrite a block of rows with N(0, (lam*sigma)^2)
    g = torch.Generator().manual_seed(seed)
    Wd = W.detach().float().cpu().clone()
    sigma = Wd.std()
    Wd[rows] = torch.randn(len(rows), Wd.shape[1], generator=g) * lam * sigma
    return Wd.to(W.dtype)

def outlier_ratio(W):
    # The 27B model's diagnostic: largest |w| / median |w|
    a = W.detach().float().abs().flatten()
    if a.numel() > 2_000_000:  # median on a random subsample for speed
        idx = torch.randint(0, a.numel(), (2_000_000,), generator=torch.Generator().manual_seed(0))
        med = a[idx].median()
    else:
        med = a.median()
    return float(a.max() / med.clamp_min(1e-12))

def clip_outliers(W, factor=15.0):
    # The 27B model's repair rule: clip every weight beyond factor x median magnitude
    Wf = W.detach().float()
    a = Wf.abs().flatten()
    med = a[torch.randint(0, a.numel(), (min(a.numel(), 2_000_000),), generator=torch.Generator().manual_seed(0))].median()
    t = factor * med
    return Wf.clamp(-t, t).to(W.dtype), float((Wf.abs() > t).float().mean())

# --- toy: 10 random "output matrices", damage one ---
torch.manual_seed(1)
mats = {f"layer{i}.down_proj": torch.randn(256, 512) * 0.011 for i in range(10)}
mats["layer4.down_proj"] = damage_matrix(mats["layer4.down_proj"], rows=list(range(100, 108)))
for k, W in mats.items(): print(f"{k:18s} max/median = {outlier_ratio(W):8.1f}")
fixed, frac = clip_outliers(mats["layer4.down_proj"])
print(f"after clip: max/median = {outlier_ratio(fixed):.1f}, share of weights clipped = {frac:.4%}")
""")

md(r"""
### 2.6 Gated DeltaNet's time-step bias (the session-8 edit)

📄 Qwen3.5 layers "mix Gated DeltaNet linear attention with full attention" (Sec. 3.1): in each group of four layers, three are linear attention and one is full attention. A linear-attention layer keeps a recurrent memory state $S_t$ and **decays** it at every token by a gate $e^{g_t}$. In the Hugging Face implementation (`modeling_qwen3_5.py`):

$$g_t = -\exp(A_{\log})\cdot \mathrm{softplus}(a_t + b_{dt}),\qquad S_t = e^{g_t}\,S_{t-1} + \dots$$

where $a_t$ is computed from the token and $b_{dt}$ is the parameter `dt_bias` (one value per head).

📄 The largest gain in the paper (+6.6 points, session 8) "comes from one uniform change: it raises the time-step bias of every linear-attention layer by 0.5". In the model's own account, this "makes these layers decay faster, so that its long reasoning gets stuck less" (Sec. 3.6, App. E.9). The code below shows numerically why raising $b_{dt}$ means faster decay (a shorter memory).
""")

code(r"""
import torch.nn.functional as F
A_log = torch.log(torch.tensor([0.01, 0.1, 1.0]))      # three illustrative heads (slow, medium, fast)
a_t = torch.tensor(-3.0)                                 # an illustrative input-dependent term
for shift in (0.0, 0.5):
    dt_bias = torch.ones(3) + shift
    g = -A_log.exp() * F.softplus(a_t + dt_bias)
    keep = g.exp()                                       # fraction of memory kept per token
    half_life = torch.log(torch.tensor(0.5)) / g         # tokens until memory halves
    print(f"dt_bias shift {shift:+.1f}: kept/token = {keep.numpy().round(4)}, half-life (tokens) = {half_life.numpy().round(3)}")
""")

md(r"""
## 3. A real Qwen3.5, and the paper's edits replayed on it

### Loading the model
🧪 **Memory trick.** The paper edits **output matrices** (`down_proj`, `out_proj`, `o_proj`) and small vectors such as `dt_bias`. We quantize every *other* linear layer to 4-bit NF4 and keep those output matrices in 16-bit (`llm_int8_skip_modules`). That way edits are real, direct parameter edits, as in the paper, while the 9B model still fits one GPU. 🧪 We load the **text-only** model (`AutoModelForCausalLM`), so the vision tower is not loaded. The paper's image results are therefore out of scope here.
""")

code(r"""
import time, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token

OUTPUT_MATRIX_KEYS = ["down_proj", "out_proj", "o_proj"]
kwargs = dict(dtype=DTYPE)
if LOAD_4BIT:
    from transformers import BitsAndBytesConfig
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=DTYPE, llm_int8_skip_modules=["lm_head"] + OUTPUT_MATRIX_KEYS)
    kwargs["device_map"] = {"": 0}
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs).eval()
if not LOAD_4BIT: model.to(DEVICE)

cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
LAYER_TYPES = cfg.layer_types; N_LAYERS = cfg.num_hidden_layers
EOS_IDS = model.generation_config.eos_token_id
EOS_IDS = set(EOS_IDS if isinstance(EOS_IDS, (list, tuple)) else [EOS_IDS]) | {tok.convert_tokens_to_ids("<|im_end|>")}
PARAMS = dict(model.named_parameters())
EDITABLE = {k: tuple(p.shape) for k, p in PARAMS.items()
            if p.dtype in (torch.float16, torch.bfloat16, torch.float32) and not any(s in k + "." for s in FORBIDDEN_SUBSTRINGS)
            and k != "model.norm.weight"}

def output_matrices(layer):
    # 📄 "the output matrices of a layer": attention-block output + MLP output
    att = f"model.layers.{layer}.linear_attn.out_proj.weight" if LAYER_TYPES[layer] == "linear_attention" \
          else f"model.layers.{layer}.self_attn.o_proj.weight"
    return [att, f"model.layers.{layer}.mlp.down_proj.weight"]

n_params = sum(p.numel() for p in model.parameters())
print(f"loaded {MODEL_ID} in {time.time()-t0:.0f}s | {N_LAYERS} layers "
      f"({LAYER_TYPES.count('linear_attention')} linear-attention, {LAYER_TYPES.count('full_attention')} full) | "
      f"stored params: {n_params/1e9:.2f}B | editable tensors: {len(EDITABLE)}")
if HAS_GPU: print(f"GPU memory allocated: {torch.cuda.memory_allocated()/2**30:.1f} GiB")
print("layer 0 output matrices:", output_matrices(0))
assert all(n in PARAMS for l in range(N_LAYERS) for n in output_matrices(l))
""")

md(r"""
### Data: practice, reserved and held-out problems
📄 The paper uses 3,000 practice problems from **DeepScaleR**, after removing anything that overlaps MATH-500, AIME 2024/25, HMMT Feb 2025, AMC 2023, OlympiadBench or GPQA-Diamond, plus 1,000 held-out problems from the same pool that no session sees. 🧪 We sample small, disjoint splits from `agentica-org/DeepScaleR-Preview-Dataset`. We keep problems with short numeric answers so grading is reliable, and we skip the benchmark-overlap filtering. If the download fails, a small built-in set is used.
""")

code(r"""
import random, json
def load_problems(seed=0):
    try:
        from datasets import load_dataset
        ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
        rows = [dict(id=f"dsr-{i}", question=r["problem"], answer=r["answer"], solution=r["solution"]) for i, r in enumerate(ds)
                if re.fullmatch(r"-?\d{1,4}", r["answer"].strip()) and len(r["problem"]) < 400]
    except Exception as e:
        print("dataset download failed, using built-in problems:", e)
        base = [("What is 17 times 23?", "391"), ("What is the sum of the first 20 positive integers?", "210"),
                ("How many positive divisors does 36 have?", "9"), ("What is 2^10 - 1000?", "24"),
                ("If 3x + 7 = 25, what is x?", "6"), ("What is the remainder when 1000 is divided by 7?", "6"),
                ("How many primes are less than 30?", "10"), ("What is 15% of 240?", "36")]
        rows = [dict(id=f"local-{i}", question=q, answer=a, solution=f"The answer is {a}.") for i, (q, a) in enumerate(base * 30)]
    random.Random(seed).shuffle(rows)
    return rows

ALL = load_problems()
need = 3 * max(N_DIAG, N_PAIRED) + N_HELDOUT
PRACTICE = ALL[: 2 * max(N_DIAG, N_PAIRED)]                      # the session may use these
RESERVED = ALL[2 * max(N_DIAG, N_PAIRED): 3 * max(N_DIAG, N_PAIRED)]  # protocol-2 check: unseen by the session
HELDOUT  = ALL[3 * max(N_DIAG, N_PAIRED): need]                  # final evaluation only
BY_ID = {p["id"]: p for p in ALL[:need]}
print(len(ALL), "candidate problems |", len(PRACTICE), "practice,", len(RESERVED), "reserved,", len(HELDOUT), "held-out")
print("example:", {k: PRACTICE[0][k] for k in ("id", "question", "answer")})
""")

md(r"""
### The generation engine (`generate` with weights = base / edited / both)
📄 Answer settings (Tab. 5): thinking on; the prompt is the question followed by *"Please reason step by step, and put your final answer within \boxed{}."*; temperature 0.6, top-k 20, top-p 0.95; budget $L$ tokens.
""")

code(r"""
SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."
def build_prompt(q):
    return tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)

def answer_problems(problems, edits=(), n=1, max_new_tokens=None, tag="gen"):
    # One answer per (problem, sample). Returns rows with text, grade, cut, loop, length.
    L = max_new_tokens or MAX_NEW_TOKENS
    rows = []
    jobs = [(p, j) for p in problems for j in range(n)]
    with use_weights(model, *edits):
        for s in range(0, len(jobs), BATCH_SIZE):
            batch = jobs[s:s + BATCH_SIZE]
            prompts = [build_prompt(p["question"]) for p, _ in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            torch.manual_seed(stable_seed(*[f"{p['id']}#{j}" for p, j in batch]))  # same seed for both weights
            out = model.generate(**enc, max_new_tokens=L, do_sample=True, temperature=0.6, top_k=20, top_p=0.95,
                                 pad_token_id=tok.pad_token_id)
            for (p, j), seq in zip(batch, out[:, enc["input_ids"].shape[1]:].tolist()):
                ended = next((i for i, t in enumerate(seq) if t in EOS_IDS), None)
                gen = seq if ended is None else seq[:ended]
                text = tok.decode(gen, skip_special_tokens=True)
                cut = ended is None
                rows.append(dict(id=p["id"], sample=j, text=text, gold=p["answer"], grade=grade(text, p["answer"]),
                                 cut=int(cut), loop=int(is_loop(text, cut)), n_tokens=len(gen)))
    return rows

def summarize(rows, name=""):
    n = len(rows); fin = [r for r in rows if not r["cut"]]
    s = dict(acc=sum(r["grade"] for r in rows) / n, cut=sum(r["cut"] for r in rows) / n,
             loop_of_cut=(sum(r["loop"] for r in rows) / max(1, sum(r["cut"] for r in rows))),
             acc_of_finished=(sum(r["grade"] for r in fin) / len(fin)) if fin else float("nan"),
             mean_tokens=sum(r["n_tokens"] for r in rows) / n)
    if name: print(f"{name:28s} acc {s['acc']:.1%} | cut {s['cut']:.1%} (looping {s['loop_of_cut']:.0%} of cut) | "
                   f"acc of finished {s['acc_of_finished']:.1%} | mean tokens {s['mean_tokens']:.0f}")
    return s

def paired_test(problems, edit, base=None, n=1, max_new_tokens=None, verbose=True):
    # Alg. 3: unedited (theta or the damaged copy) vs edited, same problems and seeds
    r0 = answer_problems(problems, edits=(base,), n=n, max_new_tokens=max_new_tokens)
    r1 = answer_problems(problems, edits=(base, edit), n=n, max_new_tokens=max_new_tokens)
    group = lambda rows, key: [[r[key] for r in rows if r["id"] == p["id"]] for p in problems]
    st = paired_stats(group(r0, "grade"), group(r1, "grade"))
    cut = paired_stats(group(r0, "cut"), group(r1, "cut"))
    st.update(acc0=summarize(r0)["acc"], acc1=summarize(r1)["acc"], cut0=summarize(r0)["cut"], cut1=summarize(r1)["cut"],
              cut_delta=cut["delta"], p_sign=sign_test_one_sided(st["n_up"], st["n_down"]))
    if verbose:
        print(f"accuracy {st['acc0']:.1%} -> {st['acc1']:.1%} (delta {st['delta']:+.3f} ± {st['se']:.3f} SE) | "
              f"cut {st['cut0']:.1%} -> {st['cut1']:.1%} | wrong->right {st['n_up']}, right->wrong {st['n_down']} | "
              f"one-sided sign test p = {st['p_sign']:.3f}")
    return st, r0, r1
""")

md(r"""
### 3.1 Starting diagnosis
📄 Before the first turn, "a starting diagnosis answers $N_0$ practice problems and reports how the model fails: its accuracy, the answers that run out of tokens or end in repetition, and how often its finished answers are right" (Sec. 2.2).

🧪 Our token budget is much smaller than the paper's 16,384, so expect **many cut answers**. That is the same failure the 27B model diagnosed in itself (answers that never finish), only more of them.
""")

code(r"""
t0 = time.time()
DIAG = answer_problems(PRACTICE[:N_DIAG])
diag = summarize(DIAG, f"starting diagnosis (N0={N_DIAG})")
print(f"({time.time()-t0:.0f}s)")
ex = next((r for r in DIAG if r["cut"]), DIAG[0])
print("\n--- tail of one answer (cut=%d) ---\n" % ex["cut"], ex["text"][-500:])
""")

md(r"""
### 3.2 Read the hidden states: finished vs. cut answers
📄 A **read** "returns the hidden states of the weights on texts of its choice". The 27B model compared its hidden states on answers that finish with those that run out of tokens, saw that "the differences grow with depth", and took the direction between the two groups (Fig. 13).

💭 *Interpretation:* the paper does not say how token positions were pooled in the model's own read. We mean-pool each answer's hidden states over its answer tokens. Below we compute, at every layer, the distance between the two group means relative to the hidden-state norm.
""")

code(r"""
import numpy as np
def read_hidden(texts, edits=(), layers=None, max_tokens=None, prompt_lens=None):
    # Mean-pooled hidden state after each chosen layer, per text -> tensor [n_texts, n_layers, d]
    layers = list(range(N_LAYERS)) if layers is None else layers
    out = []
    with use_weights(model, *edits):
        for i, t in enumerate(texts):
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_tokens or READ_MAX_TOKENS).to(model.device)
            o = model(**enc, output_hidden_states=True, use_cache=False, logits_to_keep=1)
            start = min(prompt_lens[i], enc["input_ids"].shape[1] - 1) if prompt_lens else 0
            out.append(torch.stack([o.hidden_states[l + 1][0, start:].float().mean(0).cpu() for l in layers]))
    return torch.stack(out)

def full_text(row):
    return build_prompt(BY_ID[row["id"]]["question"]) + row["text"]

fin = [r for r in DIAG if not r["cut"]]; cut_rows = [r for r in DIAG if r["cut"]]
if len(fin) < 2 or len(cut_rows) < 2:
    # 🧪 fallback when one group is (almost) empty: split by length instead
    srt = sorted(DIAG, key=lambda r: r["n_tokens"]); h = len(srt) // 2
    fin, cut_rows = srt[:h], srt[h:]
    print("not enough finished/cut answers: using shortest vs longest halves instead")
k = min(len(fin), len(cut_rows), 12)
texts = [full_text(r) for r in fin[:k] + cut_rows[:k]]
plens = [len(tok(build_prompt(BY_ID[r["id"]]["question"]))["input_ids"]) for r in fin[:k] + cut_rows[:k]]
t0 = time.time()
H = read_hidden(texts, prompt_lens=plens)            # [2k, N_LAYERS, d]
H_fin, H_cut = H[:k], H[k:]
gap = [(float((H_cut[:, l].mean(0) - H_fin[:, l].mean(0)).norm() / H[:, l].norm(dim=-1).mean())) for l in range(N_LAYERS)]
print(f"read {2*k} texts in {time.time()-t0:.0f}s")
for l in range(N_LAYERS):
    print(f"layer {l:2d} ({LAYER_TYPES[l][:4]}): relative gap {gap[l]:.3f} " + "#" * int(60 * gap[l] / max(gap)))
""")

md(r"""
### 3.3 Edit A: remove the "unfinished" direction at an early layer
📄 The 27B model first edited a late layer, where the difference was largest. Its tests said the edit hurt, so it dropped it. It then moved to **layer 8 of 64** and removed 5% of the direction (finished → cut) from that layer's two output matrices. "Each of its three paired tests of this edit favours it" (Sec. 3.3, Fig. 13).

🧪 We replay the *same recipe* at the *same relative depth* (8/64 = 12.5% of the depth) on our model. This is not the model discovering the edit (that is Part 4). It answers a narrower question: does the edit the 27B model wrote for itself transfer to a smaller sibling? Try different layers and $\alpha$ in the exercises.
""")

code(r"""
def make_direction_edit(layer, alpha, H_a=H_cut, H_b=H_fin):
    u = mean_diff_direction(H_a[:, layer], H_b[:, layer])     # from finished (b) to cut (a)
    edit = {}
    for name in output_matrices(layer):
        W = PARAMS[name].detach().clone()
        W_new = remove_direction(W.float(), u.to(W.device), alpha).to(W.dtype).cpu()
        assert check_edit(name, W.cpu(), W_new) is None
        edit[name] = W_new
        print(f"  {name}: relative change {float((W_new.float() - W.float().cpu()).norm() / W.float().norm()):.4%}")
    return edit

LAYER_A = round(N_LAYERS * 8 / 64)
print(f"Edit A at layer {LAYER_A} ({LAYER_TYPES[LAYER_A]}), alpha = 0.05")
EDIT_A = make_direction_edit(LAYER_A, 0.05)
test_ids = PRACTICE[N_DIAG:N_DIAG + N_PAIRED] if len(PRACTICE) >= N_DIAG + N_PAIRED else PRACTICE[:N_PAIRED]
stA, _, _ = paired_test(test_ids, EDIT_A)
""")

md(r"""
### 3.4 Edit B: time-step bias +0.5 in every linear-attention layer
📄 Session 8's edit, written by the 27B model: "time-step bias +0.5 in all 48 linear-attention layers". Result on 490 held-out problems × 8 answers: 57.2% → 63.8% (+6.6, 95% interval [5.0, 8.1], p < 10⁻⁵), and answers cut at the limit fell from 33.6% to 26.1% (Tab. 13, App. E.9).
""")

code(r"""
EDIT_B = {}
for l, t in enumerate(LAYER_TYPES):
    if t == "linear_attention":
        name = f"model.layers.{l}.linear_attn.dt_bias"
        W = PARAMS[name].detach().cpu().clone()
        EDIT_B[name] = (W.float() + 0.5).to(W.dtype)
        assert check_edit(name, W, EDIT_B[name]) is None
print(f"Edit B touches {len(EDIT_B)} dt_bias vectors")
stB, _, _ = paired_test(test_ids, EDIT_B)
""")

md(r"""
**Check on held-out problems.** 📄 An edit counts only if it improves problems that were not used to choose it (Eq. 2). 🧪 With only a few dozen problems the standard error is large, so treat any single toy result as anecdotal. The paper's evaluation used 490 problems × 8 answers.
""")

code(r"""
best_name, best_edit = max([("A (direction, layer %d)" % LAYER_A, EDIT_A, stA), ("B (dt_bias +0.5)", EDIT_B, stB)],
                           key=lambda t: t[2]["delta"] - 0.5 * t[2]["cut_delta"])[:2]
print("held-out check of edit", best_name)
st_held, _, _ = paired_test(HELDOUT, best_edit)
""")

md(r"""
### 3.5 Toy experiment: damage the model, then repair it
📄 Sec. 2.4 and 3.2: damage 8 consecutive rows of the MLP output matrix of one layer **at the same relative depth** (layer 30 of 64) with $\lambda=200$ noise. Diagnose it by comparing largest and median weights across output matrices. Repair it by clipping at 15× the median.

This is the most reliable effect in this notebook, and it works even on the 0.8B model: one corrupted block silently breaks the model, and a simple statistic finds it.
""")

code(r"""
DMG_LAYER = round(N_LAYERS * 30 / 64)
dmg_name = f"model.layers.{DMG_LAYER}.mlp.down_proj.weight"
W_clean = PARAMS[dmg_name].detach().cpu().clone()
r0 = random.Random(0).randrange(0, W_clean.shape[0] - 8)
DMG_ROWS = list(range(r0, r0 + 8))
DAMAGE = {dmg_name: damage_matrix(W_clean, DMG_ROWS, lam=200.0, seed=0)}   # the record of this is NOT shown to the editor
print(f"damaged {dmg_name}, rows {DMG_ROWS[0]}-{DMG_ROWS[-1]}")

dmg_rows = answer_problems(HELDOUT, edits=(DAMAGE,))
summarize(dmg_rows, "damaged copy (held-out)")
print("damaged answer tail:", repr(dmg_rows[0]["text"][-200:]))
""")

code(r"""
# Diagnosis as the 27B model did it: largest / median weight of every output matrix, on the DAMAGED copy
with use_weights(model, DAMAGE):
    census = sorted(((outlier_ratio(PARAMS[n]), n) for l in range(N_LAYERS) for n in output_matrices(l)), reverse=True)
for r, n in census[:5]: print(f"{r:9.1f}x  {n}")
print(f"... the next highest after the top one: {census[1][0]:.1f}x")
named = census[0][1]
print("named matrix:", named, "| correct:", named == dmg_name)

with use_weights(model, DAMAGE):
    W_dmg = PARAMS[named].detach().cpu().clone()   # clone: on CPU, .cpu() would alias the live parameter
W_rep, frac = clip_outliers(W_dmg, 15.0)
REPAIR = {named: W_rep}
assert check_edit(named, W_dmg, W_rep) is None
print(f"clipped {frac:.4%} of the matrix's weights")
print("largest |w| in damaged rows -> original %.3f | damaged %.2f | repaired %.3f" % (
    W_clean[DMG_ROWS].float().abs().max(), W_dmg[DMG_ROWS].float().abs().max(), W_rep[DMG_ROWS].float().abs().max()))
corr = np.corrcoef(W_clean[DMG_ROWS].float().flatten().numpy(), W_rep[DMG_ROWS].float().flatten().numpy())[0, 1]
print(f"correlation of repaired vs original values in damaged rows: {corr:+.3f}  (📄 the paper: uncorrelated; the repair caps, it does not restore)")
""")

md(r"""
🧪 **A cheap extra signal (not in the paper): next-token loss on reference solutions.** Accuracy at a small token budget is noisy. The mean next-token loss of each set of weights on a few held-out reference solutions shows the damage and the repair in seconds.
""")

code(r"""
def mean_nll(texts, edits=(), max_tokens=256):
    tot = n = 0
    with use_weights(model, *edits):
        for t in texts:
            enc = tok(t, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
            out = model(**enc, labels=enc["input_ids"])
            k = enc["input_ids"].shape[1] - 1
            tot += float(out.loss) * k; n += k
    return tot / n
REF = [p["question"] + "\n\n" + p["solution"] for p in HELDOUT[:4]]
for name, e in [("base", ()), ("damaged", (DAMAGE,)), ("repaired", (DAMAGE, REPAIR))]:
    print(f"{name:9s} mean next-token loss on reference solutions: {mean_nll(REF, e):.3f}")
""")

code(r"""
rep_rows = answer_problems(HELDOUT, edits=(DAMAGE, REPAIR))
base_rows = answer_problems(HELDOUT)
summarize(base_rows, "base model (held-out)")
summarize(dmg_rows,  "damaged copy (held-out)")
summarize(rep_rows,  "repaired copy (held-out)")
n_up = sum(a["grade"] == 0 and b["grade"] == 1 for a, b in zip(dmg_rows, rep_rows))
n_dn = sum(a["grade"] == 1 and b["grade"] == 0 for a, b in zip(dmg_rows, rep_rows))
print(f"damaged -> repaired: wrong->right {n_up}, right->wrong {n_dn}, sign test p = {sign_test_one_sided(n_up, n_dn):.4f}")
""")

md(r"""
## 4. Main method: a minimal self-rewriting session (Alg. 1 and 2)

Here the model **is its own editor** ($\phi=\theta$). 📄 Alg. 1, at each turn: the editor replies with a short rationale and **exactly one** action:
- **code**: Python on a CPU, with a persistent namespace, gradients disabled, and refusal of optimizers, model loading, processes and network access;
- **generate**: answer chosen practice problems with weights `"base"`, `"edited"` or `"both"` (a paired test, which costs one forward pass);
- **read**: hidden states of the weights on chosen texts (one forward pass);
- a **final answer** with no block ends the session, and whatever edit is saved at that point is submitted.

Budgets are $B$ forward passes, $C$ code runs and $T_{\max}$ turns (the model is not told $T_{\max}$). 📄 In the **second protocol** (Fig. 1), the framework tests the kept edit once on **reserved** practice problems the session never saw, and the model then submits or withdraws.

🧪 Our tool interface is a compact paraphrase of App. D.4. The model writes fenced blocks: ```` ```python ```` for code, and ```` ```generate ```` / ```` ```read ```` containing a JSON object. Our Python sandbox is a best-effort `exec` with a keyword blocklist, **not a security boundary**. Run it only in a disposable runtime such as Colab.

📄 Expectation: the paper's 9B session "perturb[ed] [itself] at random and [broke itself]" (Fig. 2). The 4B and 9B models added `np.random.randn(...) * 0.01` to their weights (Fig. 11). If you see the same here, you have reproduced the paper's scale finding. The guard against harm is the paired test plus the reserved check: a bad edit gets withdrawn.
""")

code(r"""
import io, contextlib, signal, traceback, textwrap

TASK = textwrap.dedent(f'''
You are {MODEL_ID}, a language model with {N_LAYERS} layers ({LAYER_TYPES.count("linear_attention")} Gated DeltaNet
linear-attention layers and {LAYER_TYPES.count("full_attention")} full-attention layers). The weights you will edit
are YOUR OWN; this conversation runs on them. Your job: become better at competition mathematics by changing
those weights directly. Nothing will train you: no gradients, no optimizers, no training of any kind.

How you will be judged: after the session, your weights with and without your edit answer unseen math problems
(thinking on, budget {MAX_NEW_TOKENS} tokens). Only the final \\boxed{{}} answer after the thinking is graded;
an answer whose thinking is still open when the budget runs out earns no credit.

STARTING DIAGNOSIS on {N_DIAG} practice problems: {{diag}}

TOOLS. Reply each turn with a few sentences on what the last result showed and what comes next, then EXACTLY ONE block:
(1) ```python ... ``` runs on a CPU in a persistent session (gradients off). Available names:
    torch, np, TENSORS (dict name->shape of editable tensors), load(name) -> float32 copy of a tensor (unedited),
    save_edit(name, W), drop_edit(name), list_edits(), output_matrices(layer) -> [attention_out, mlp_down],
    unit(v), mean_diff(A, B) -> unit vector from mean(B) to mean(A), remove_direction(W, u, alpha) = W - alpha*u u^T W,
    DIAG (list of graded answers: id, grade, cut, loop, n_tokens, text), PRACTICE_IDS, GEN (results of generate
    passes by number), READ (results of read passes by number: tensor [n_texts, n_layers, hidden]).
(2) ```generate {{"weights": "both", "ids": [...], "max_new_tokens": {MAX_NEW_TOKENS}}}``` answers practice problems
    with your weights without the edit ("base"), with it ("edited"), or both on the same seeds (paired test).
(3) ```read {{"source": "diag", "which": "cut" or "finished" or "all", "layers": [0, 4, 8], "weights": "base"}}```
    runs your weights over those answers and stores mean hidden states after the chosen layers in READ.
You have {SESSION_PASSES} forward passes (generate or read) and {SESSION_CODE_RUNS} code runs. Failed requests cost nothing.

HOW TO WORK. Look at your unfinished and wrong answers first; find the components behind the failure; test edits
against your unedited self on the same problems; keep an edit only if the paired test supports it. Out of scope: the
embedding, the final norm, the output layer, and multiplying a whole tensor by one number.
AIM: clearly fewer answers lost to the failure in the diagnosis, without lower accuracy. If no edit does that, submit none.
When your forward passes run out, reply with a final answer that contains no block: describe your edit and the
evidence. Whatever is saved as your edit at that point is submitted; to withdraw, drop every edit first.
''').strip()

BLOCKED = ("backward", "optim", "requires_grad", "from_pretrained", "subprocess", "os.system", "socket", "requests",
           "urllib", "import os", "import sys", "__import__", "open(", "exec(", "eval(", "shutil", "pipeline(", "generate(")

class Session:
    def __init__(self, base=None, diag_rows=None, passes=None, code_runs=None, turns=None):
        self.B = passes or SESSION_PASSES; self.C = code_runs or SESSION_CODE_RUNS; self.T = turns or SESSION_TURNS
        self.base = base                       # None (own weights) or a damage edit (repair session)
        self.edit = {}                         # Δ: the current edit
        self.b = self.c = self.t = 0
        self.diag = diag_rows
        self.gen, self.read = {}, {}
        self.ns = self._namespace()
        self.log = []

    def _clean(self, name):
        with use_weights(model, self.base):
            return PARAMS[name].detach().float().cpu().clone()

    def _namespace(self):
        S = self
        def load(name):
            if name not in EDITABLE: raise KeyError(f"{name} is not an editable tensor")
            return S._clean(name)
        def save_edit(name, W):
            if name not in EDITABLE: return f"refused: {name} is not editable"
            W = torch.as_tensor(W).float()
            why = check_edit(name, S._clean(name), W)
            if why: return why
            S.edit[name] = W.to(PARAMS[name].dtype); return f"saved edit for {name}"
        def drop_edit(name): S.edit.pop(name, None); return f"dropped {name}"
        public_diag = [{k: r[k] for k in ("id", "grade", "cut", "loop", "n_tokens")} | {"text": r["text"][-1500:]} for r in S.diag]
        return dict(torch=torch, np=np, TENSORS=EDITABLE, load=load, save_edit=save_edit, drop_edit=drop_edit,
                    list_edits=lambda: list(S.edit), output_matrices=output_matrices, unit=unit,
                    mean_diff=mean_diff_direction, remove_direction=remove_direction, DIAG=public_diag,
                    PRACTICE_IDS=[p["id"] for p in PRACTICE], GEN=S.gen, READ=S.read, print=print)

    def run_python(self, src):
        if any(k in src for k in BLOCKED):
            return "refused: the code uses a blocked name (gradients, optimizers, model loading, processes, files or network)"
        buf = io.StringIO()
        def _timeout(*a): raise TimeoutError("code run stopped after 600 s")
        use_alarm = hasattr(signal, "SIGALRM")
        try:
            if use_alarm: signal.signal(signal.SIGALRM, _timeout); signal.alarm(600)
            with torch.no_grad(), contextlib.redirect_stdout(buf):
                exec(src, self.ns)
        except Exception:
            buf.write(traceback.format_exc(limit=2))
        finally:
            if use_alarm: signal.alarm(0)
        return buf.getvalue()[:10000] or "(no output)"

    def forward(self, kind, req):
        if kind == "generate":
            ids = [i for i in req.get("ids", []) if i in BY_ID and i in {p["id"] for p in PRACTICE}][:N_PAIRED]
            if not ids: ids = [p["id"] for p in PRACTICE[:N_PAIRED]]
            probs = [BY_ID[i] for i in ids]
            L = int(min(max(16, int(req.get("max_new_tokens", MAX_NEW_TOKENS))), MAX_NEW_TOKENS))
            w = req.get("weights", "both")
            k = len(self.gen)
            if w == "both":
                st, r0, r1 = paired_test(probs, self.edit, base=self.base, max_new_tokens=L, verbose=False)
                self.gen[k] = dict(base=r0, edited=r1, stats=st)
                return (f"GEN[{k}] paired test on {len(ids)} problems: accuracy {st['acc0']:.1%} -> {st['acc1']:.1%} "
                        f"(delta {st['delta']:+.3f} ± {st['se']:.3f}); cut {st['cut0']:.1%} -> {st['cut1']:.1%}; "
                        f"wrong->right {st['n_up']}, right->wrong {st['n_down']}")
            rows = answer_problems(probs, edits=(self.base, self.edit) if w == "edited" else (self.base,), max_new_tokens=L)
            self.gen[k] = {w: rows}
            s = summarize(rows)
            return f"GEN[{k}] {w} weights on {len(ids)} problems: accuracy {s['acc']:.1%}, cut {s['cut']:.1%}, mean tokens {s['mean_tokens']:.0f}"
        if kind == "read":
            which = req.get("which", "all")
            rows = [r for r in self.diag if which == "all" or (which == "cut") == bool(r["cut"])][:16]
            if not rows: raise ValueError(f"no diagnosis answers with which={which}")
            layers = [int(l) for l in req.get("layers", list(range(N_LAYERS))) if 0 <= int(l) < N_LAYERS] or list(range(N_LAYERS))
            plens = [len(tok(build_prompt(BY_ID[r["id"]]["question"]))["input_ids"]) for r in rows]
            edits = (self.base, self.edit) if req.get("weights") == "edited" else (self.base,)
            Hs = read_hidden([full_text(r) for r in rows], edits=edits, layers=layers, prompt_lens=plens)
            k = len(self.read); self.read[k] = Hs
            norms = Hs.norm(dim=-1).mean(0).tolist()
            return f"READ[{k}] = tensor {tuple(Hs.shape)} ({len(rows)} '{which}' answers x layers {layers}); mean norm per layer: " + \
                   ", ".join(f"L{l}:{n:.1f}" for l, n in zip(layers, norms))
        raise ValueError(kind)

BLOCK_RE = re.compile(r"```(python|generate|read)\s*\n?(.*?)```", re.S)

def editor_reply(messages):
    # The editor is the model itself, on its own unedited weights (phi = theta): no edit is loaded here.
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=EDITOR_MAX_NEW_TOKENS, do_sample=True, temperature=0.6, top_k=20, top_p=0.95,
                         pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    think, ans, closed = split_thinking(text)
    return ans.strip() if closed and ans.strip() else text.strip()   # if thinking never closed, look for blocks anyway

def run_session(sess, verbose=True, editor=None):
    editor = editor or editor_reply
    d = summarize(sess.diag)
    diag_txt = (f"accuracy {d['acc']:.1%}; {d['cut']:.1%} of answers ran out of tokens ({d['loop_of_cut']:.0%} of those "
                f"end in repetitive text); finished answers are right {d['acc_of_finished']:.1%} of the time.")
    task = TASK.replace("{diag}", diag_txt).replace(f"You have {SESSION_PASSES} forward passes", f"You have {sess.B} forward passes")
    if sess.base is not None:
        task += ("\n\nDAMAGED COPY. The weights you edit in this session are a copy of your own weights in which some values "
                 "have been corrupted, as a memory or hardware fault might do. You are not told which, where, or how. load(), "
                 "the diagnosis and every forward pass refer to this damaged copy. You yourself run on your undamaged weights.")
    messages = [{"role": "user", "content": task}]
    final = False
    while True:
        sess.t += 1
        if sess.t > sess.T and not final:
            messages.append({"role": "user", "content": "time is up: reply now with your final answer (no code)"}); final = True
        t0 = time.time(); reply = editor(messages); dt = time.time() - t0
        blocks = BLOCK_RE.findall(reply)
        if verbose: print(f"\n===== turn {sess.t} ({dt:.0f}s) =====\n{reply[:2500]}")
        # 📄 Alg. 2: the session ends on a reply after "time is up", or a block-free reply once all B passes are used
        if final or (not blocks and sess.b >= sess.B):
            sess.log.append(("final", reply)); break
        if len(blocks) != 1:
            obs = "reply with exactly one block (```python, ```generate or ```read)"
        else:
            kind, body = blocks[0]
            if kind == "python":
                if sess.c >= sess.C: obs = "no code runs left"
                else: obs = sess.run_python(body); sess.c += 1
            elif sess.b >= sess.B:
                obs = "no forward passes left: reply with your final answer"
            else:
                try:
                    obs = sess.forward(kind, json.loads(body)); sess.b += 1
                except Exception as e:
                    obs = f"request failed (not counted): {e}"
        obs += f"\n[forward passes used {sess.b} of {sess.B}; code runs {sess.c} of {sess.C}; turn {sess.t}; current edit: {list(sess.edit) or 'none'}]"
        if verbose: print(f"----- observation -----\n{obs[:2000]}")
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": obs}]
        if sess.b >= sess.B and sess.c >= sess.C: final = True
    return sess.edit
print("session machinery ready")
""")

md(r"""
### 4.0 Dry run: a scripted editor goes through the same tools
Before the real model takes over, a **scripted** editor (🧪 not a model) replays the steps the paper reports for the 27B model's first-protocol session (📄 Fig. 13): read the hidden states, compute the direction between finished and unfinished answers, remove 5% of it from an early layer's output matrices, run a paired test, then give a final answer. This checks that the tool plumbing works and shows what a well-formed session looks like.
""")

code(r"""
SCRIPT = [
    f'Step 1: read my hidden states on my diagnosis answers.\n```read\n{{"which": "all", "layers": [{LAYER_A}]}}\n```',
    "Step 2: direction from finished to unfinished answers; remove 5% of it from both output matrices.\n```python\n"
    "H = READ[0][:, 0]\n"
    "cut = torch.tensor([r['cut'] for r in DIAG][:H.shape[0]]).bool()\n"
    "if cut.all() or (~cut).all():\n"
    "    order = torch.tensor([r['n_tokens'] for r in DIAG][:H.shape[0]]).argsort()\n"
    "    cut = torch.zeros_like(cut); cut[order[len(order)//2:]] = True\n"
    "u = mean_diff(H[cut], H[~cut])\n"
    f"for name in output_matrices({LAYER_A}):\n"
    "    print(save_edit(name, remove_direction(load(name), u, 0.05)))\n```",
    f'Step 3: paired test of the edit against my unedited self.\n```generate\n{{"weights": "both", "max_new_tokens": {MAX_NEW_TOKENS}}}\n```',
    "Final answer: I removed 5% of the finished-to-unfinished direction from the two output matrices of an early layer; "
    "the paired test above is the evidence.",
]
def scripted_editor(messages, _it=iter(SCRIPT)):
    return next(_it, SCRIPT[-1])
DRY = Session(base=None, diag_rows=DIAG, passes=2, code_runs=2, turns=6)
DRY_EDIT = run_session(DRY, editor=scripted_editor)
print("\ndry-run edit:", list(DRY_EDIT), "| turns", DRY.t, "| passes", DRY.b, "| code runs", DRY.c)
assert DRY.b == 2 and DRY.c == 1 and len(DRY_EDIT) == 2
""")

md(r"""
### 4.1 Run a session: the model edits its own weights
This can take a while: every editor turn is up to `EDITOR_MAX_NEW_TOKENS` of generation, and every `generate` pass is a paired test. Set `RUN_SESSION = 0` in the config to skip it.
""")

code(r"""
if RUN_SESSION:
    t0 = time.time()
    SESSION = Session(base=None, diag_rows=DIAG)
    SUBMITTED = run_session(SESSION)
    print(f"\nsession finished in {(time.time()-t0)/60:.1f} min: turns {SESSION.t}, passes {SESSION.b}, code runs {SESSION.c}")
    print("edit on disk:", {k: tuple(v.shape) for k, v in SUBMITTED.items()} or "none (the model withdrew)")
else:
    SUBMITTED = {}
""")

md(r"""
### 4.2 Protocol 2: check on reserved problems, then submit or withdraw, then evaluate on held-out
📄 In the second protocol, "before the model submits an edit, the framework tests it once on reserved practice problems that the session can neither read nor use, and the model then submits the edit or withdraws it" (Sec. 2.2). 🧪 Here a simple rule stands in for the model's decision: submit only if the check does not lower accuracy and does not raise the cut share.
""")

code(r"""
if SUBMITTED:
    print("reserved-problem check:")
    st_chk, _, _ = paired_test(RESERVED[:N_PAIRED], SUBMITTED)
    if st_chk["delta"] >= 0 and st_chk["cut_delta"] <= 0:
        print("-> submit. Held-out evaluation (never seen by the session):")
        st_final, _, _ = paired_test(HELDOUT, SUBMITTED)
    else:
        print("-> withdraw: the model in use stays as it was (as in sessions 5 and 6 of the paper's Tab. 13)")
else:
    print("No edit submitted: the model withdrew or made no edit (📄 the paper's 2B model made no edit; in five of "
          "eight protocol-2 sessions even the 27B model withdrew).")
""")

md(r"""
### 4.3 (Optional) A repair session: the clean model repairs its damaged copy
📄 The editor runs on its **undamaged** weights, while its diagnosis, every forward pass and every weight it reads come from the damaged copy (Sec. 2.4). Uncomment to try it. Compare whatever your model does with the 27B model's census and clipping in 3.5.
""")

code(r"""
# dmg_diag = answer_problems(PRACTICE[:N_DIAG], edits=(DAMAGE,))
# REPAIR_SESSION = Session(base=DAMAGE, diag_rows=dmg_diag)
# REPAIR_EDIT = run_session(REPAIR_SESSION)
# if REPAIR_EDIT: paired_test(HELDOUT, REPAIR_EDIT, base=DAMAGE)
""")

md(r"""
## 5. Limitations: how this toy differs from the paper

| | 📄 Paper | 🧪 This notebook |
|---|---|---|
| Model | Qwen3.5 2B/4B/9B/**27B**, bf16, served with vLLM | Qwen3.5-**9B** (4-bit NF4, output matrices in 16-bit) or 0.8B, HF `generate` |
| Answer budget | 16,384 tokens (4,096 in some 27B sessions) | 1,024 (9B) / 384 (small) by default, so far more answers are cut |
| Session budgets | B=20 passes, C=200 code runs, T<sub>max</sub>=160, 2.5–3.5 h, replies up to 12,000 tokens | B=4, C=30, T<sub>max</sub>=16, replies up to 2,048 tokens |
| Diagnosis / tests | 256 (or 64) diagnosis problems; up to 256 answers per side per paired test | 8–24 problems |
| Evaluation | 200 held-out problems at 16,384 tokens; 490 × 8 answers in a pre-registered paired test | a few dozen held-out problems, so the noise is large |
| Seeds | one seed per problem (hash of id) | one seed per batch (hash of the batch's ids), identical for both weights |
| Tools | read returns hidden states, MLP activations or token log-probs on any texts; notes file; context reset beyond 90k tokens | read returns mean-pooled hidden states of diagnosis answers only; no notes file or context reset |
| Sandbox | CPU Python session that refuses gradients, model loading, processes and network | keyword blocklist plus `exec`: **not** a security boundary |
| Damage | layer 30/64 of 27B, rows 3068–3075, λ=200 (chosen so weaker noise did not break it, Tab. 6) | same relative depth and λ, random rows; we did not tune λ for our model |
| Images | the repair carries over to six image benchmarks | text-only model; images not covered |

Further caveats:
- **The edits in Part 3 were written by the 27B model, not by ours.** Replaying them on a 9B or 0.8B model tests transfer, not self-rewriting. They may not help a smaller model or a tiny token budget, and that is a valid result.
- 💭 Mean-pooling over answer tokens, the "same relative depth" mapping of layer indices, and the submit/withdraw rule in 4.2 are my interpretations.
- Generation is not exactly reproducible, even with fixed seeds (📄 App. D.3 says the same about their setup). Single pairs prove nothing, which is why the paper uses sign and permutation tests over many problems.
- The paper's conclusion stresses that self-rewriting should keep **people in the loop**: every self-made model is evaluated, for safety as well as the task, and confirmed by a person before use.
""")

md(r"""
## 6. Exercises

1. **Layer sweep (Sec. 3.3).** Re-run Edit A at every 4th layer with α ∈ {0.02, 0.05, 0.2}. Does an early layer beat a late one, as the 27B model found? Plot δ̂ ± SE per layer.
2. **Session 2's edit (Tab. 13).** The 27B model also submitted an edit that "subtracts from the two output matrices of layer 63 five times their projection on one direction" (α = 5 at the last layer). Implement it with `remove_direction(W, u, 5.0)` at the last layer and run a paired test. Why might α > 1 still work?
3. **Decay sweep (Sec. 2.6).** Try dt_bias shifts of −0.5, +0.25, +0.5, +1.0 and +3.0 (📄 session 1 used +3.0 in 7 layers and gave +0.5 points, not significant). Plot accuracy and cut share against the shift.
4. **Token budget.** Raise `MAX_NEW_TOKENS` to 4,096. Does Edit B's effect shrink as fewer answers are cut?
5. **Damage strength (Tab. 6).** Repeat 3.5 with λ ∈ {20, 60, 200}. 📄 For the 27B model only 200σ broke it (69.8% base; 70.8%, 66.7% and 0.0% for 20σ, 60σ and 200σ). Where is the threshold for your model?
6. **Ablation (Fig. 15).** Clip all output matrices at 15× the median versus only the named one. Do the extra clips cost accuracy, as in the paper (55.0% for the named matrix alone vs. 33.0% for all 20)?
7. **Scale (Sec. 3.8).** Run the Part 4 session with `MODEL_SIZE="small"`, then `"9B"`. Classify each session by the steps in Fig. 2: does its code run, does it run itself, test its edits, read its states, and act on its tests?
8. **Sign test vs. SE.** For your held-out results, compute both the SE-based z-score from Eq. (3) and the sign-test p-value. When do they disagree?
""")

md(r"""
## 7. Appendix: numbers quoted from the paper

| Result | Value | Source |
|---|---|---|
| 27B self-edit, protocol 2, session 8 (dt_bias +0.5, all 48 linear-attention layers) | 57.2% → **63.8%** on 490 held-out × 8; 95% CI [5.0, 8.1]; p < 10⁻⁵; cut 33.6% → 26.1% | Fig. 1, Tab. 13 |
| 27B self-edit, protocol 2, session 2 (layer 63 direction, ×5) | 57.2% → 61.3% (+4.0, p < 10⁻⁵); cut 28.8% | Tab. 13 |
| 27B self-edit, protocol 2, session 1 (dt_bias +3.0 in 7 layers) | 57.2% → 57.8% (+0.5, p = 0.25) | Tab. 13 |
| Protocol 2 sessions 3–7 | withdrew (3, 4, 7: no edit kept; 5, 6: withdrew after the check) | Tab. 13 |
| 27B self-edit, protocol 1 (layer 8, α = 0.05) | own task 22.4% → 29.6%; held-out 57.2% → 58.3% (+1.1, p = 0.038) | Tab. 1, Tab. 2 |
| 27B repair of damaged copy | 0.0% → 33.0% on 200 held-out (base 58.5%); named matrix alone 55.0%; other 19 alone 0.5% | Tab. 1, Fig. 15 |
| Repair on images | 7.3% → 70.2% on six benchmarks (base 84.3%); ScreenSpot-v2 0% → 89% (base 93%) | Sec. 1 |
| Damage census | damaged matrix 1397× median; no other above 162.3× | Fig. 14a |
| Scale (Tab. 2) | 2B: no edit; 4B and 9B: their repair edits leave the damaged copy at 0.0%; only 27B improves and repairs | Fig. 2, Tab. 2 |
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "accelerator": "GPU",
                  "colab": {"provenance": [], "gpuType": "L4"}}
nbf.validate(nb)
import os
nbf.write(nb, os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebook.ipynb"))
print("wrote", len(cells), "cells")
