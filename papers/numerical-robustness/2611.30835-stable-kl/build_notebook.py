import nbformat as nbf
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))
COLAB = "https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/papers/numerical-robustness/2611.30835-stable-kl/notebook.ipynb"
SHA = "4486b36d6aa644730d8bf8869bf4886e1ffeebfd"

md(rf"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

# Paper2Code guided tour: *The KL Divergence Is Never Negative, Except in PyTorch*

**Paper:** Zhen Yang, Wei Tang, Chu Chen, Rui Qian, Yixuan Chen, Fan Yang, Kangning Cui. arXiv:2611.30835 (cs.LG, 23 Nov 2026), [PDF](https://arxiv.org/pdf/2611.30835v1).

**Official code:** [github.com/xi-xiaoran/stablekl](https://github.com/xi-xiaoran/stablekl). The code is BSD-3-Clause and the data CC BY 4.0. The paper's abstract links it ("Code and data"), and its README cites the paper. It contains:
- the `stablekl` package
- the benchmark and scoring code
- every experiment script
- small result files

The 10.8 GB frozen benchmark lives on Hugging Face (`xi-xiaoran/stablekl-benchmark`) and is **not** needed here.

Following the Paper2Code rule, this tour **does not reimplement the method**. It clones the official repo, pinned to commit `{SHA[:7]}`, and installs it. Then it:
- puts every key equation next to the official source function that implements it
- runs small but real versions of the paper's experiments

**The paper's LLM** is Qwen2.5-1.5B-Instruct (Exp. 5.4). Section 9 uses it in FP16 on GPU or CPU, with a `SMALL=1` switch to Qwen2.5-0.5B-Instruct. A 9B model is not needed, because the paper's own model is small enough to run as-is.

**Labels:** 📄 stated in the paper or repo · 🧪 a simplification or choice made in this notebook · 💭 my interpretation. Numbers marked 📄 are quoted from the PDF. Every other number is computed live in this notebook.
""")

md(r"""
## Contents
0. Setup: clone and install the official repo
1. The idea in plain English
2. The bug in one number: an Exponential KL that PyTorch returns as 0
3. Eq. (1) and (3): φ(x) = x − log1p(x) and its series, next to `ExponentialKLC`
4. A mini Figure 1: relative error and negative-KL rate against the true KL
5. Eq. (2): the Normal KL and its analytic gradient, next to `NormalKLC`
6. Interface B (`F.kl_div`): operator error vs input-representation error
7. Eq. (4): the logits path, next to `LogitsKL`
8. Downstream experiments, small versions: 8a Geometric fit (Exp. 5.1, official script), 8b Bayesian design's Beta case (Exp. 5.2), 8c identical-start distillation (Exp. 5.3, official script)
9. The LLM: Qwen2.5-1.5B-Instruct vs an 8-bit RTN copy (Exp. 5.4 Part A)
10. Cost, limitations, exercises, and the paper's numbers
""")

md("## 0. Setup")
code(rf"""
import os, sys, subprocess, inspect, math, time, warnings, textwrap
REPO_URL = "https://github.com/xi-xiaoran/stablekl.git"
PIN = os.environ.get("STABLEKL_COMMIT", "{SHA}")   # 🧪 pinned for reproducibility; set to "" for the latest main
REPO = os.path.abspath("stablekl-repo")   # hyphenated so the clone folder is not importable as a package
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO], check=True)
if PIN:
    subprocess.run(["git", "-C", REPO, "checkout", "-q", PIN], check=True)
print(subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %ad %s"], capture_output=True, text=True).stdout)
try:
    import stablekl  # already installed (e.g. local run)
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", REPO], check=True)
    import importlib; importlib.invalidate_caches(); sys.path.insert(0, REPO)
    import stablekl
for sub in ("experiments/common", "experiments/exp1_geometric_fit", "experiments/exp3_identical_distillation", "experiments/exp4a_quantized_eval"):
    sys.path.insert(0, os.path.join(REPO, sub))
""")
code(r"""
import numpy as np, torch, torch.nn.functional as F, mpmath as mp
import matplotlib.pyplot as plt
from torch.distributions import Exponential, Normal, Beta, Categorical, kl_divergence
warnings.simplefilter("ignore")
mp.mp.dps = 60
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("stablekl", stablekl.__version__, "| torch", torch.__version__, "| device", DEV)
print("📄 the paper evaluated PyTorch 2.15.0 (CPU) and 2.4.0 (A100). Other versions may differ in individual bits, not in the picture.")

def show(obj, start=None, stop=None, title=None):
    # print the official source of a function/class (optionally a line range of it)
    src = inspect.getsource(obj).splitlines()
    f = inspect.getsourcefile(obj); l0 = inspect.getsourcelines(obj)[1]
    f = os.path.join(REPO, f[f.rindex("stablekl" + os.sep):]) if "stablekl" + os.sep in f else f
    a, b = (start or 0), (stop or len(src))
    print(f"# {os.path.relpath(f, REPO)}  lines {l0 + a}-{l0 + b - 1}" + (f"  ({title})" if title else ""))
    print("\n".join(src[a:b]))
""")

md(r"""
## 1. The idea in plain English

📄 The KL divergence between two distributions is never negative and is zero only when they are equal. PyTorch computes it with textbook closed forms. For the Exponential family that form is KL = r − 1 − log r with r = λq/λp. When P and Q are close (relative difference δ), the true KL is of order δ². The formula, however, subtracts terms of order 1 and δ. Once the KL falls to about the unit roundoff u of the format, the result is mostly rounding noise:
- FP32: u = 2⁻²⁴ ≈ 6×10⁻⁸
- FP16: u ≈ 5×10⁻⁴

That noise can make the result negative, exactly zero, or wrong in sign for the gradient.

📄 The paper measures this on 8.5M inputs across PyTorch's two KL interfaces:
- **interface A:** `torch.distributions.kl_divergence`, with 18 closed forms
- **interface B:** `F.kl_div`, the loss behind `nn.KLDivLoss`

📄 Near KL ≈ 10⁻⁷ in FP32 (10⁻³ in FP16), over 85% of values are off by at least 1%. At smaller divergences about 30% are negative.

📄 Simply running the official formula one precision higher ("OFFICIAL+") still leaves 7–51% significantly wrong in 22 of 23 configurations.

📄 The repair rewrites each formula as a sum of terms that are non-negative by construction, such as φ(x) = x − log1p(x). It also:
- computes the arguments as exactly computed differences of the inputs
- implements analytic gradients
- in strict mode ("OURS"), evaluates in FP64 and rounds **once** to the target format

📄 For distillation and LLM evaluation there is a separate **logits path**. It computes KL(softmax(a) ‖ softmax(c)) straight from the logits, without ever forming rounded probabilities.

💭 This is about operator correctness, not modelling. It matters in exactly the places where the KL is supposed to be small:
- a student close to its teacher
- a quantized model close to the original
- a posterior that barely moves
- a fit at its optimum
""")

md(r"""
## 2. The bug in one number

📄 The paper's opening example uses λp = 1 and λq = fl(1.0001), the FP32 number nearest 1.0001. Official FP32 code returns **0**, but the true KL is **5.0013×10⁻⁹**. Below, the true value comes from mpmath at 60 digits, applied to the *same* FP32 inputs.
""")
code(r"""
def kl_exp_true(lp, lq):
    r = mp.mpf(float(lq)) / mp.mpf(float(lp)); return r - 1 - mp.log(r)

lp = torch.tensor([1.0]); lq = torch.tensor([1.0001])          # FP32
P, Q = Exponential(lp), Exponential(lq)
print("official torch.distributions.kl_divergence :", kl_divergence(P, Q).item())
print("stablekl.kl_divergence (mode='ours')        :", stablekl.kl_divergence(P, Q).item())
print("stablekl.kl_divergence (mode='rewrite')     :", stablekl.kl_divergence(P, Q, mode="rewrite").item())
print("true KL of the same FP32 inputs (mpmath)    :", mp.nstr(kl_exp_true(lp, lq), 8))
""")

md(r"""
## 3. Eq. (1) and (3) next to the official source

📄 **Eq. (1):** φ(x) = x − log1p(x) ≥ 0 for x > −1. This is the Exponential KL with x = (λq − λp)/λp, and it is the building block of almost every other family.

📄 **Eq. (3):** φ itself still cancels as x → 0. So on −¾ ≤ x ≤ 3 the code uses log1p(x) = 2 atanh(s) with s = x/(2+x):

  φ(x) = x²/(2+x) − 2 s³ Σ_{k≥1} s^{2(k−1)}/(2k+1),

truncated after 9 / 17 / 36 terms for FP16 / FP32 / FP64. 📄 The truncation error is bounded by u/16 relative to φ.

In `ExponentialKLC.forward` below, map the code to the equations as follows:
- `x = d / a`, with `d = b − a`: x is computed from the *difference* of the inputs. That subtraction is exact for nearby floats (Sterbenz).
- `t1 = xs*xs/tx`: the x²/(2+x) term.
- `t2 = 2*(s*z)*q`: 2s³·Σ, with the Horner loop over `KTERMS`.
- `clamp_min(·, 0)`: the result can never be negative.
- outside the segment, x − log1p(x) is used directly.
- `gp = −x/a`, `gq = x/b`: the analytic gradients.
""")
code(r"""
from stablekl._closed import exponential as EXP
print("KTERMS:", EXP.KTERMS, " segment:", (EXP.SEG_LO, EXP.SEG_HI))
show(EXP.ExponentialKLC.forward)
""")
code(r"""
# Check eq. (3) as implemented against mpmath, in FP32, and compare with the official formula r - 1 - log r.
xs = torch.tensor([1e-7, 1e-5, 1e-3, 0.1, -0.5, 2.5], dtype=torch.float32)
a = torch.ones_like(xs); b = a + xs                       # λp = 1, λq = 1 + x (rounded to FP32)
ours = EXP.ExponentialKLC.apply(a, b)
off = kl_divergence(Exponential(a), Exponential(b))
print(f"{'x (FP32)':>12} {'true φ':>14} {'stablekl':>14} {'rel.err':>9} {'official':>14} {'rel.err':>9}")
for i in range(len(xs)):
    x = mp.mpf(float(b[i])) - 1; t = x - mp.log1p(x)
    re_o = float(abs(mp.mpf(float(ours[i])) - t) / t); re_f = float(abs(mp.mpf(float(off[i])) - t) / t)
    print(f"{float(b[i]-1):12.3e} {mp.nstr(t, 6):>14} {float(ours[i]):14.6e} {re_o:9.1e} {float(off[i]):14.6e} {re_f:9.1e}")
print("identical inputs -> ", EXP.ExponentialKLC.apply(a, a.clone()).tolist(), "(exact +0)")
""")

md(r"""
## 4. A mini Figure 1

📄 Figure 1 of the paper plots relative error against the true KL. 🧪 Here we draw a small random sample for Exponential and Normal: λq = λp(1+δ), and for Normal both σ and μ are perturbed by δ. We compare three methods against mpmath:
- **OFFICIAL:** PyTorch in the input format
- **OFF.+:** the official formula evaluated one format higher, then rounded
- **OURS:** stablekl strict mode

🧪 This is not the paper's stratified 8.5M-input benchmark (that is the HF dataset). It is a few thousand points meant to show the shape of the curves.
""")
code(r"""
N_PER = int(os.environ.get("FIG1_N", "60"))
deltas = np.logspace(-9, -0.5, 26)
up = {torch.float16: torch.float32, torch.float32: torch.float64}
def kl_norm_true(pm, ps, qm, qs):
    pm, ps, qm, qs = (mp.mpf(float(v)) for v in (pm, ps, qm, qs))
    return mp.log(qs / ps) + (ps**2 + (pm - qm)**2) / (2 * qs**2) - mp.mpf(1) / 2

def sample(fam, dt, delta, n, g):
    if fam == "Exponential":
        lp = torch.exp(torch.randn(n, generator=g, dtype=torch.float64)).to(dt)
        lq = (lp.double() * (1 + delta * torch.randn(n, generator=g, dtype=torch.float64))).to(dt)
        return (lp, lq), lambda T: (Exponential(lp.to(T)), Exponential(lq.to(T))), lambda i: kl_exp_true(lp[i], lq[i])
    ps = torch.exp(0.3 * torch.randn(n, generator=g, dtype=torch.float64)).to(dt); pm = torch.randn(n, generator=g, dtype=torch.float64).to(dt)
    qs = (ps.double() * (1 + delta * torch.randn(n, generator=g, dtype=torch.float64))).to(dt)
    qm = (pm.double() + delta * ps.double() * torch.randn(n, generator=g, dtype=torch.float64)).to(dt)
    return None, lambda T: (Normal(pm.to(T), ps.to(T)), Normal(qm.to(T), qs.to(T))), lambda i: kl_norm_true(pm[i], ps[i], qm[i], qs[i])

rows = []
g = torch.Generator().manual_seed(0)
for fam in ("Exponential", "Normal"):
    for dt in (torch.float32, torch.float16):
        for d in deltas:
            _, mk, truth = sample(fam, dt, d, N_PER, g)
            P, Q = mk(dt); off = kl_divergence(P, Q).double()
            P2, Q2 = mk(up[dt]); offp = kl_divergence(P2, Q2).to(dt).double()
            ours = stablekl.kl_divergence(P, Q).double()
            for i in range(N_PER):
                t = truth(i)
                if not (t > 0) or (dt == torch.float16 and t < 6e-5):   # 📄 FP16 rel. error shown only while the true KL is normal
                    continue
                rows.append((fam, str(dt)[-7:], float(t), *[float(abs(mp.mpf(float(v[i])) - t) / t) for v in (off, offp, ours)], float(off[i]) < 0))
import pandas as pd
df = pd.DataFrame(rows, columns=["family", "dtype", "true_kl", "OFFICIAL", "OFF.+", "OURS", "official_negative"])
df["bin"] = np.floor(np.log10(df.true_kl)).astype(int)
fig, axes = plt.subplots(1, 4, figsize=(18, 3.6), sharey=True)
for ax, ((fam, dtn), sub) in zip(axes, df.groupby(["family", "dtype"])):
    med = sub.groupby("bin")[["OFFICIAL", "OFF.+", "OURS"]].median()
    for col, st in zip(med.columns, ("o-", "s--", "^-")):
        ax.plot(10.0 ** med.index, med[col].clip(lower=1e-17), st, label=col)
    ax.set(xscale="log", yscale="log", title=f"{fam} {dtn}", xlabel="true KL"); ax.grid(alpha=.3)
axes[0].set_ylabel("median relative error"); axes[0].legend(); plt.tight_layout(); plt.show()
summ = df.assign(sig_1pct=df.OFFICIAL >= 0.01).groupby(["family", "dtype", "bin"])[["sig_1pct", "official_negative"]].mean().mul(100).round(1)
print("% of official values off by >=1% and % negative, per decade of true KL (this sample):")
print(summ.unstack(["family", "dtype"]).to_string())
print("max relative error of OURS over all", len(df), "points:", f"{df.OURS.max():.2e}", "(FP16 unit roundoff is 4.9e-4, FP32 6.0e-8)")
print("max relative error of OURS, FP32 points only:", f"{df[df.dtype == 'float32'].OURS.max():.2e}")
""")

md(r"""
💭 In this small sample the two families' official values go **wrong by being rounded to zero or too small, not by turning negative**. The negative values that the paper counts (📄 about 30% at small divergences, across all families) show up below in Beta (§8b), `F.kl_div` (§7, §9) and the Geometric fit (§8a).

## 5. Eq. (2): the Normal KL, next to `NormalKLC`

📄 For P = N(μp, σp²) and Q = N(μq, σq²), let e = (σp − σq)/σq and w = (μp − μq)/σq. Then

  **KL(P‖Q) = φ(e) + ½e² + ½w²**  (2),

whereas the official code evaluates ½(r² + w² − 1 − log r²) with r = σp/σq, "in which w² is absorbed by r² ≈ 1". 📄 The analytic gradient is ∂KL/∂σp = e(e+2)/σp.

Where to find the pieces in the source:
- `e = ds / qs`, with `ds = ps − qs` an exact difference
- `phiS = es*s − 2*(s*z)*q`: the same φ series as eq. (3), with s = (σp−σq)/(σp+σq)
- `KLs = phi + ef*half*ef`: φ(e) + ½e²
- `KLm = w*half*w`: ½w²
- `g_ps = Pe/ps`, with `Pe = e*(e+2)`: e(e+2)/σp
""")
code(r"""
from stablekl._closed import normal as NOR
show(NOR.NormalKLC.forward, 0, 60)
""")
code(r"""
pm, ps, qm, qs = (torch.tensor([v], dtype=torch.float32) for v in (0.0, 1.0001, 1e-4, 1.0))
P, Q = Normal(pm, ps), Normal(qm, qs)
t = kl_norm_true(pm, ps, qm, qs)
print("true KL (mpmath, same FP32 inputs):", mp.nstr(t, 8))
print("official                          :", kl_divergence(P, Q).item())
print("stablekl ours / rewrite           :", stablekl.kl_divergence(P, Q).item(), stablekl.kl_divergence(P, Q, mode="rewrite").item())
# gradient wrt sigma_p: autograd of the official formula vs stablekl's analytic gradient vs mpmath
for name, f in (("official autograd", lambda a, b: kl_divergence(Normal(pm, a), Normal(qm, b))),
                ("stablekl analytic", lambda a, b: stablekl.kl_divergence(Normal(pm, a), Normal(qm, b)))):
    a = ps.clone().requires_grad_(); f(a, qs).sum().backward(); print(f"dKL/dsigma_p {name:18s}: {a.grad.item(): .6e}")
e = (mp.mpf(float(ps)) - mp.mpf(float(qs))) / mp.mpf(float(qs))
print("dKL/dsigma_p true e(e+2)/sigma_p   :", mp.nstr(e * (e + 2) / mp.mpf(float(ps)), 7))
# identical distributions, different-looking parameters are not needed: identical inputs return exact zeros
a = ps.clone().requires_grad_(); L = stablekl.kl_divergence(Normal(pm, a), Normal(pm.clone(), ps.clone())); L.backward()
print("P == Q -> stablekl KL, grad:", L.item(), a.grad.item(), "| official:", kl_divergence(Normal(pm, ps), Normal(pm, ps.clone())).item())
""")

md(r"""
## 6. Interface B: `F.kl_div`

📄 `F.kl_div(input, target)` takes the student's **log**-probabilities and the teacher's probabilities. The paper separates two error sources:
1. **Operator error:** the elementwise target·(log target − input) and its reduction are computed inaccurately. `stablekl.kl_div` fixes this. In `mode="ours"` it is correctly rounded on all 83,899,496 forward and 483,279,156 backward scored components of the benchmark, against 52.5% / 74.9% for the official code (📄 Sec. 4).
2. **Input-representation error:** `log_softmax(c)` and `softmax(a)` were already rounded *before* the call. No operator can recover what was lost. 📄 This is why the LLM experiments use the logits path (Sec. 7).

🧪 Below, the teacher logits are close to the student's (a = c + 10⁻³·noise, V = 1000), and the inputs are FP32.
""")
code(r"""
torch.manual_seed(0)
c = torch.randn(4, 1000); a = c + 1e-3 * torch.randn(4, 1000)
lq, p = c.log_softmax(-1), a.softmax(-1)
ref_logits = F.kl_div(c.double().log_softmax(-1), a.double().softmax(-1), reduction="none").sum(-1)        # FP64 truth from the logits
ref_inputs = F.kl_div(lq.double(), p.double(), reduction="none").sum(-1)                                   # FP64 truth of the rounded inputs
print(pd.DataFrame({"true (from logits)": ref_logits.numpy(),
                    "true (of rounded inputs)": ref_inputs.numpy(),
                    "F.kl_div official": F.kl_div(lq, p, reduction="none").sum(-1).double().numpy(),
                    "stablekl.kl_div": stablekl.kl_div(lq, p, reduction="none").sum(-1).double().numpy()}).to_string(float_format="%.4e"))
print("\nbatchmean:  official", F.kl_div(lq, p, reduction="batchmean").item(), "| stablekl.kl_div", stablekl.kl_div(lq, p, reduction="batchmean").item(),
      "| logits_kl", stablekl.logits_kl(a, c).item(), "| truth", ref_logits.mean().item())
print("reduction='mean' (divides by B*V, a common mistake):", F.kl_div(lq, p, reduction="mean").item())
""")
md(r"""
💭 How to read the table:
- `stablekl.kl_div` matches "true (of rounded inputs)", so the operator is fixed.
- Both columns still differ from "true (from logits)" by several percent, because the information was lost when the probabilities were rounded.
- `logits_kl` matches the logits truth.

📄 The README notes that in rare cases `kl_div(mode="ours")` raises `UncertifiedRounding` rather than guessing, when even its interval evaluation cannot decide the rounding.
""")

md(r"""
## 7. Eq. (4): the logits path, next to `LogitsKL`

📄 For teacher logits a and student logits c: let q = softmax(c) in FP64, and let δj be the centred differences of a − c. Then

  Δ = log1p(Σj qj expm1(δj)),  wj = δj − Δ,  KL = Σj qj (wj e^{wj} − expm1(wj)),  ∂KL/∂cj = −qj expm1(wj).  (4)

📄 Here pj = qj e^{wj} is the teacher probability, and h(w) = w eʷ − expm1(w) ≥ 0. Near 0, h is evaluated by its series Σ_{n≥2} (n−1)wⁿ/n!.

Mapping to `LogitsKL`:
- `u = A − C`
- `S = Σ q·expm1(u)`, `delta = log1p(S)`, `w = u − delta`
- `terms = q·_phi(w)`
- in the backward pass, `g = −(G/B)·q·expm1(w)`

`_phi` is h(w). Its coefficients `(k+1)/(k+2)!` are the series (n−1)/n! with n = k+2.

💭 About centring:
- The paper centres δ at the student's arg-max. The code uses the plain FP64 difference u, and **centres only in the guarded rows** (`_guarded`): those where the FP64 subtraction was inexact (`lo != 0`), 1+S cancels, or something overflows.
- Mathematically, Δ absorbs any common offset, so both give the same KL.

📄 The output is rounded **once** from FP64 by `_round_once`. That function rounds to FP32 with round-to-odd before FP16/BF16, which avoids double rounding.
""")
code(r"""
from stablekl import logits as LG
show(LG._phi); print(); show(LG.LogitsKL.forward); print(); show(LG.LogitsKL.backward)
""")
code(r"""
# h(w) series vs mpmath; and the two structural zeros: identical logits, and logits shifted by a constant
for w in (1e-6, 1e-3, 0.05, -0.08, 0.5, -3.0):
    tw = mp.mpf(w); t = tw * mp.e**tw - mp.expm1(tw)
    print(f"w={w:8.0e}  h true {mp.nstr(t, 10):>18}  _phi {LG._phi(torch.tensor([w], dtype=torch.float64)).item():.10e}")
torch.manual_seed(1)
V = 151936                                                     # 📄 Qwen2.5 vocabulary size used in Exp. 5.3
t_log = (torch.round(3 * torch.randn(8, V) * 64) / 64).clamp(-12, 12).half()   # 🧪 on a 1/64 grid so that adding 4 is exact in FP16
assert ((t_log + 4).float() - t_log.float() == 4).all()
for name, s_log in (("identical", t_log.clone()), ("shifted by +4", t_log + 4)):
    for meth in ("F.kl_div", "Categorical", "logits_kl"):
        s = s_log.clone().float().requires_grad_()
        sd = s.half()
        if meth == "F.kl_div":      L = F.kl_div(sd.log_softmax(-1), t_log.softmax(-1), reduction="batchmean")
        elif meth == "Categorical": L = kl_divergence(Categorical(logits=t_log), Categorical(logits=sd)).mean()
        else:                       L = stablekl.logits_kl(t_log, sd)
        L.backward()
        print(f"FP16 V={V} {name:14s} {meth:12s} loss={L.item(): .3e}  |grad|={s.grad.norm().item():.3e}")
""")
md(r"""
💭 With identical (or shifted) logits the true KL and gradient are exactly zero. Whatever the official losses return here (negative, positive, or ∞ / NaN) is the spurious signal that Exp. 5.3 feeds to an optimiser.
""")

md(r"""
## 8. Downstream experiments, small versions

### 8a. Exp. 5.1: fitting a Geometric distribution (official `geo_fit.py`)
📄 **Setup:** Adam (lr 10⁻²), 3,000 steps, minimising the KL to a target p⋆.
📄 **Result:** in FP16 with p⋆ = 0.001, the official loss reaches the target from 0 of 12 starts and the repair from all 12. At the optimum itself the official code returns a non-zero loss or gradient in 28 of 32 settings, for example loss 0.098 and gradient −100 in FP16 with p⋆ = 0.001. The repair returns exact zeros in all 32.

🧪 We call the script's own `run()` and `opt_check()`:
- `official` = torch.distributions
- `C` = the stablekl rewrite in the same format
- `D` = strict: FP64, rounded once

We use one start (the script's default p0 = min(0.5, 3p⋆)) and a few settings.
""")
code(r"""
import geo_fit as GF
STEPS = int(os.environ.get("GEO_STEPS", "3000"))
res = []
for dt in (torch.float16, torch.float32):
    for param in ("probs", "logits"):
        for pstar in (0.3, 0.001):
            for method in ("official", "C", "D"):
                r = GF.run(method, param, "rev", pstar, dt, steps=STEPS)
                res.append(dict(dtype=str(dt)[-7:], param=param, pstar=pstar, method=method, final_rel_err=r["final_rel_err"],
                                loss_lt0=r["neg"], loss_eq0=r["zero"], nonfinite=r["nonfinite"], loss_at_opt=r["loss_at_opt"], grad_at_opt=r["grad_at_opt"]))
geo = pd.DataFrame(res); print(geo.to_string(float_format="%.3g"))
""")

md(r"""
### 8b. Exp. 5.2: the Beta case behind wrong arm choices
📄 For Beta(n+1, n) vs Beta(n, n), the true KL is 2.5×10⁻⁴, 2.5×10⁻⁵ and 2.5×10⁻⁶ for n = 10³, 10⁴, 10⁵. The official FP32 code returns −1.5×10⁻³, −5.6×10⁻³ and +0.31.

📄 The experiment runs K = 10 Bernoulli arms and picks the arm with the largest expected information gain. The official code chooses a wrong arm in 40–91% of rounds in 7 of 8 settings; the rewrite makes no wrong choice.

🧪 Here we only reproduce the three KL values. The full loop is `experiments/exp2_beta_design/beta_ig.py`.
""")
code(r"""
def kl_beta_true(a1, b1, a2, b2):
    a1, b1, a2, b2 = map(mp.mpf, (a1, b1, a2, b2))
    lB = lambda a, b: mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)
    return lB(a2, b2) - lB(a1, b1) + (a1 - a2) * mp.digamma(a1) + (b1 - b2) * mp.digamma(b1) + (a2 - a1 + b2 - b1) * mp.digamma(a1 + b1)
for n in (1e3, 1e4, 1e5):
    N = torch.tensor(n); P, Q = Beta(N + 1, N), Beta(N, N)
    print(f"n={n:.0e}  true {mp.nstr(kl_beta_true(n+1, n, n, n), 5):>10}   official FP32 {kl_divergence(P, Q).item(): .4e}   stablekl {stablekl.kl_divergence(P, Q).item(): .4e}")
""")

md(r"""
### 8c. Exp. 5.3: identical-start distillation (official `distill_gpu.py`)
📄 **Setup:**
- The teacher is an MLP 32 → 256 → V. The student is an exact copy of it.
- Training runs 300 steps on fresh inputs.
- The true KL and its gradient are exactly zero throughout, so any movement of the student is caused by numerical error.

📄 **Results:**
- The student moves in all 2,160 runs with the official losses, and in none of the 720 runs with the repair.
- Table 2, Adam lr 10⁻², top-1 disagreement after 300 steps (official F.kl_div / ours; median over 10 seeds): **V = 32,768 FP16: 41.0 / 0**, FP32: 40.5 / 0; **V = 151,936 FP16: 97.4 / 0**.

🧪 We call the script's `run()` with one seed. The default is V = 32,768 on CPU and V = 151,936 on a GPU (override with `DISTILL_V`). `logitsA` is the logits path (eq. 4).
""")
code(r"""
import distill_gpu as DG
V = int(os.environ.get("DISTILL_V", "151936" if DEV.type == "cuda" else "32768"))
SEEDS = [int(s) for s in os.environ.get("DISTILL_SEEDS", "0").split(",")]
METHODS = os.environ.get("DISTILL_METHODS", "kl_div,categorical,logitsA").split(",")
out = []
for m in METHODS:
    for sd in SEEDS:
        t0 = time.time(); r = DG.run(DEV, V, "fp16", m, "adam", sd, steps=300, lr=1e-2)
        last = r["log"].get(300, r["log"][max(r["log"])] if r["log"] else {})
        out.append(dict(method=m, seed=sd, status=r["status"], loss0=r.get("loss0"), gnorm0=r.get("gnorm0"),
                        rel_dW=last.get("rel_dW"), true_kl=last.get("true_kl"), top1_disagree_pct=100 * last.get("top1_disagree", float("nan")), sec=round(time.time() - t0, 1)))
        print(out[-1], flush=True)
print(pd.DataFrame(out).to_string(float_format="%.3g"))
""")

md(r"""
## 9. The LLM: Qwen2.5-1.5B-Instruct vs its 8-bit RTN copy (Exp. 5.4, Part A)

📄 **Setup:** KL(original ‖ quantized), per token, on 16,384 WikiText-2 tokens (32 windows × 512) in FP16.

📄 **Result** (Table 3, 8-bit, GPTQ / RTN):
- official `F.kl_div`: mean error −19.0% / −4.4%, with 30.2% / 12.3% of tokens negative
- `reduction="mean"` reports 0
- `Categorical` returns ∞ (about 38% of FP16 probabilities underflow)
- the logits path: 0.0 / 0.0, with 0 negative and 0 infinite tokens

🧪 Our version is smaller:
- **RTN only.** We call the repo's `rtn()` from `exp4a_quantized_eval/dequant.py`. GPTQ needs the separate Hugging Face checkpoint.
- **`N_WINDOWS` windows of 512 tokens** (default 2, which is 1,024 tokens).
- The reference is the repo's own `exact_kl()` from `run_eval.py`: the FP64 logits-path formula applied to the FP16 logits.

**Model switches:**
- the default is the paper's `Qwen/Qwen2.5-1.5B-Instruct`
- `SMALL=1` switches to `Qwen/Qwen2.5-0.5B-Instruct`
- `LLM_MODEL=<hf id>` loads any other model

On CPU the model still runs in FP16, as in the paper. This works in recent PyTorch and needs about 4 GB of RAM for 1.5B. Set `SKIP_LLM=1` to skip this section.
""")
code(r"""
SKIP_LLM = os.environ.get("SKIP_LLM", "0") == "1"
MODEL = os.environ.get("LLM_MODEL") or ("Qwen/Qwen2.5-0.5B-Instruct" if os.environ.get("SMALL") == "1" else "Qwen/Qwen2.5-1.5B-Instruct")
N_WINDOWS, WIN = int(os.environ.get("N_WINDOWS", "2")), 512
if not SKIP_LLM:
    try:
        import transformers, datasets
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "datasets", "accelerate"], check=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from dequant import rtn                  # official RTN quantiser
    from run_eval import exact_kl            # official FP64 reference (logits-path formula)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEV).eval()
    text = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[0][: N_WINDOWS * WIN].view(N_WINDOWS, WIN)
    @torch.no_grad()
    def logits_of(m): return torch.cat([m(w[None].to(DEV)).logits[0].float().cpu() for w in ids]).half()
    t0 = time.time(); base = logits_of(model)
    for n, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and n != "lm_head":
            mod.weight.data = rtn(mod.weight.data, 8).to(DEV)
    quant = logits_of(model); del model
    print(MODEL, "| tokens:", base.shape[0], "| vocab:", base.shape[1], f"| {time.time()-t0:.0f}s")
""")
code(r"""
if not SKIP_LLM:
    ref = torch.cat([exact_kl(base[i:i+256], quant[i:i+256]) for i in range(0, len(base), 256)])     # FP64, per token
    lq, pb, lb = quant.log_softmax(-1), base.softmax(-1), base.log_softmax(-1)
    per = {
        "F.kl_div": F.kl_div(lq, pb, reduction="none").float().sum(-1).double(),
        "F.kl_div, log_target": F.kl_div(lq, lb, reduction="none", log_target=True).float().sum(-1).double(),
        "Categorical": kl_divergence(Categorical(logits=base), Categorical(logits=quant)).double(),
        "logits path (logits_kl)": torch.stack([stablekl.logits_kl(base[i:i+1], quant[i:i+1]) for i in range(len(base))]).double(),
    }
    tab = []
    for k, v in per.items():
        fin = torch.isfinite(v)
        tab.append(dict(method=k, mean_err_pct=100 * (v.mean() / ref.mean() - 1).item(), neg_tokens_pct=100 * (v < 0).float().mean().item(),
                        inf_tokens_pct=100 * (~fin).float().mean().item()))
    print("FP64 reference mean KL per token:", f"{ref.mean().item():.4e}")
    print(pd.DataFrame(tab).to_string(float_format="%.2f"))
    print("F.kl_div(reduction='mean') reports:", F.kl_div(lq, pb, reduction="mean").item())
    print("FP16 softmax probabilities that underflow to 0 (base / quantized):", f"{100 * (pb == 0).float().mean().item():.1f}% / {100 * (lq.exp() == 0).float().mean().item():.1f}%")
    # 🧪 logits_kl rounds each token's KL to FP16 once; feeding the (exactly representable) FP16 logits as FP64 shows the unrounded value
    lk64 = torch.stack([stablekl.logits_kl(base[i:i+1].double(), quant[i:i+1].double()) for i in range(len(base))])
    print("logits path in FP64 vs reference: max |rel diff| =", f"{((lk64 - ref).abs() / ref.abs().clamp_min(1e-300)).max().item():.1e}")
    plt.figure(figsize=(7, 3.2))
    rel = ((per["F.kl_div"] - ref) / ref).clamp(-3, 3).numpy()
    plt.hist(rel, bins=80); plt.axvline(-1, color="r", ls=":", label="rel. err = −1 (KL ≤ 0 below)"); plt.legend()
    plt.title(f"official F.kl_div per-token relative error, {MODEL.split('/')[-1]} vs RTN-8 (FP16)"); plt.xlabel("relative error"); plt.show()
""")
md(r"""
💭 Expect numbers of the same kind as the 📄 RTN column (−4.4%, 12.3% negative). They will not be identical, because we use far fewer tokens and possibly a different PyTorch version and hardware. 📄 The paper says "about 38% of the FP16 softmax probabilities underflow to zero"; the share printed above is counted over every vocabulary entry of every token in this sample. 💭 The paper's count may be over a different set, so do not compare the two directly.
""")

md(r"""
## 10. Cost, limitations, exercises

**Cost** (📄 README and Sec. 4):
- `REWRITE` costs 2.4–5.9× the official call on GPU for five of seven timed families.
- Strict `ours` costs 3.4–12×; Gamma is about 30× and Beta about 320×.
- `kl_div` costs 27–59× on GPU and `logits_kl` 3–22×.
- 📄 The repairs are unoptimised; there are no fused kernels.
""")
code(r"""
n = 1_000_000
P = Normal(torch.randn(n), torch.rand(n) + 0.5); Q = Normal(torch.randn(n), torch.rand(n) + 0.5)
for name, f in (("official", lambda: kl_divergence(P, Q)), ("stablekl rewrite", lambda: stablekl.kl_divergence(P, Q, mode="rewrite")),
                ("stablekl ours", lambda: stablekl.kl_divergence(P, Q))):
    f(); t0 = time.perf_counter(); [f() for _ in range(3)]; print(f"Normal, {n:,} pairs, cpu: {name:17s} {(time.perf_counter()-t0)/3*1e3:7.1f} ms")
""")
md(r"""
**Limitations and gaps** (📄 paper Sec. 7 and the README; 💭 marks my reading):
- 📄 The benchmark deliberately oversamples small divergences. The error *rates* describe that regime, not a typical training batch.
- 📄 BF16 results are exploratory.
- 📄 Five interval-certification tests are `xfail`.
- 📄 `logits_kl` is tested against high-precision references but is not interval-certified.
- 📄 `kl_div(mode="ours")` can raise `UncertifiedRounding`.
- 📄 The frozen benchmark (10.8 GB) is on Hugging Face, and model checkpoints are not redistributed.
- 📄 Exp. 3 at full scale (the full grid, V = 151,936) and Exp. 4 need a GPU (`docs/REPRODUCE.md`).
- 💭 The fix only matters when the KL is small relative to the format's roundoff. Large divergences are computed fine by PyTorch.

**Exercises:**
1. Switch `mode="rewrite"` in Section 4 and find where it still differs from `ours` (📄 mostly gradients near zero crossings, Beta and Dirichlet).
2. Run Section 8c with `DISTILL_METHODS=kl_div,kl_div_lt,upcast64,logitsA` and see whether upcasting to FP64 inside `F.kl_div` alone removes the drift.
3. Set `SMALL=1` or `N_WINDOWS=32` (the paper's 16,384 tokens) in Section 9.
4. Run `pytest tests -m "not slow"` inside the repo (📄 Level 0 of `docs/REPRODUCE.md`).

**Numbers from the paper** (📄):
- 8.5M inputs: 7,542,469 interface-A samples (7,234,454 scored) and 942,200 interface-B batches, over 23 configurations.
- Over 85% of FP32 values are off by ≥ 1% near KL 10⁻⁷, and about 30% are negative at smaller divergences.
- OFFICIAL+ leaves 7–51% significantly wrong in 22 of 23 configurations. REWRITE removes > 97% of significant errors.
- OURS has no significant error on 7,234,454 samples and correctly rounds all 567M interface-B components.
- At KL 10⁻¹⁵, 6.8% of official gradients point into the opposite half-space.
- GPU (A100) aggregate rates are within 1.5 percentage points of CPU.
- In GPT-2 circuit discovery (Exp. 5.4 Part D), the library's FP32 edge scores are wrong for 19%, 46% and 59% of edges; the logits path gets none wrong.
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata.update(kernelspec=dict(name="python3", display_name="Python 3", language="python"), colab=dict(provenance=[]), accelerator="GPU")
nbf.write(nb, "notebook.ipynb"); print("wrote notebook.ipynb with", len(cells), "cells")
