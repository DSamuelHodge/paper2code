import os, re, nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))
FOLDER = "papers/physiological-sensing/patent-US20190120821A1-sidestream-vo2"
COLAB = f"https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/{FOLDER}/notebook.ipynb"
core = open(os.path.join(HERE, "vo2sim_core.py")).read()
parts = re.split(r"\n(?=# ---- )", core)
CORE = {i: p.strip("\n") for i, p in enumerate(parts)}

md(rf"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB})

# Paper2Code (patent edition): building a sidestream breath-by-breath VO₂ device by simulation

**Source:** US 2019/0120821 A1, *Breath Analysis Device*. Inventor Apostolos Atsalakis; applicant Endo Medical Inc. (Palo Alto, CA). Published 25 Apr 2019; PCT/US17/26979 filed 11 Apr 2017; priority GB 1606292.9 (12 Apr 2016) and EP 17386006.5 (21 Feb 2017). [PDF](https://patentimages.storage.googleapis.com/5c/e8/03/4010b9c8ca2d6f/US20190120821A1.pdf) · [Google Patents](https://patents.google.com/patent/US20190120821A1/en)

**Status as of 23 Nov 2026:** the application was **granted as US 12,736,522 B2 on 15 Sep 2026**. The assignee is now **PNOE Inc.** (Endo Medical changed its name; recorded 5 Jan 2023). Sources: [Justia](https://patents.justia.com/patent/12736522) and the Google Patents legal-event timeline (issue fee received 7 Jul 2026). The granted claim 1 is **much narrower** than the published claim 1. See section 0.

**No official code exists** (it is a patent). This notebook is a **from-scratch teaching simulator** in numpy/scipy/matplotlib. It runs on CPU in about a minute, with no GPU and no LLM.

**Labels used everywhere:**
- 📜 stated in the patent (paragraph numbers refer to the A1 publication)
- 📚 textbook physiology or physics, with the source named
- 🧪 a simulation assumption or method choice made here; the numbers are illustrative, not measurements
- 💭 my interpretation

> ⚠️ **Patent caution, not legal advice.** Building a device for your own use is a different situation from making or selling one. Before you sell, sell kits of, or publish a product based on this architecture, get a **freedom-to-operate review** from a patent attorney. It should cover US 12,736,522 and PNOE's related family. The claim notes below are a reading aid, not an opinion on infringement.
""")

md(r"""
## Contents
0. What the patent says (and what the granted claims cover)
1. Setup and the simulation core
2. Ground truth: synthetic breathing (rest, then a ramp test)
3. The sidestream: transport delay, mixing, first-order O₂/CO₂ sensors
4. Humidity and BTPS / ATPS / STPD
5. Recovering the time offsets: flow↔gas edge, and the patent's O₂-vs-mirrored-CO₂ fit (eq. 5)
6. Breath detection, then breath-by-breath VO₂/VCO₂ via Haldane, including the patent's own constants
7. RER, Weir REE and Frayn substrate oxidation
8. Slow sensors: breath skipping, sampling chamber, fast sensors, software speed-up
9. Mask leaks: why the circuit must be sealed
10. Ambient auto-zero between breaths
11. VT1/VT2: V-slope and ventilatory equivalents on the ramp
12. Design knobs: tube, pump and sensor τ, mapped to delay and breath-by-breath accuracy
13. Summary: patent vs textbook vs assumption; caveats
""")

md(r"""
## 0. What the patent says

**Architecture** (📜 [0012], [0044]–[0060], FIG. 1/2):
- A **primary pathway** (mouthpiece or facemask 1 → flow sensor 2 → T-connector 14 → outlet 3) carries the whole breath, both directions.
- A **flow sensor** sits in the primary pathway, either before the branch (FIG. 1) or after it (FIG. 2). Listed types: hot-film anemometer, micro-thermal-conductivity detector, thermal mass flow, ultrasonic transit-time ([0053]). When the branch is before the flow sensor, the reading must account for the constant sampling flow.
- A **secondary (sidestream) pathway** branches off at the T and carries, in order: Nafion® dryer 4 → micro-pump 5 → three-way valve 6 → **O₂ sensor 7 and CO₂ sensor 8 *in-line, in series*** → barometric pressure 9, RH 10 and temperature 11 sensors → optional extra sensors 13 ([0133]–[0137]).
- The patent says in-line (series) placement is better than a "parallel" sidestream that splits the sample to each sensor, because the split causes turbulence and pressure swings ([0016]). **This in-line arrangement is the core of claim 1.**
- The sample is preferably taken from the **periphery** of the primary pathway, where flow is less turbulent ([0049]).
- The pump keeps a constant flow, and draws **ambient air in the dead time between exhalations**: "Measurements of ambient air can be useful for calibration of the device" ([0072]). A rotary vane pump is preferred ([0073]). Without a pump, narrow tubes set the flow ([0074]).
- **Breath skipping** ([0076]–[0081]): at about 15 breaths/min every breath is analysed. At about 40–50 breaths/min "the response time of the sensors may be insufficient", so only every 2nd, 3rd or 4th breath "or even fewer" is analysed. The microcontroller picks the mode from the breathing rate.
- **Sensor types listed.**
  - O₂ ([0055]): electrochemical, paramagnetic, fuel cell, fluorescence quenching (ruthenium dye), micro-TCD, metal oxide, polymer. "A sensor with a response rate in the range of milliseconds is especially beneficial."
  - CO₂ ([0056]): NDIR, metal oxide, potentiometric solid state, micro-TCD, polymer.
  - A TCD can stand in for one gas if the other is measured and N₂ is taken as 78.08 % ([0058]). CNT arrays could sense both ([0060]).
- **Humidity** ([0063]–[0070]): Nafion tube(s) bring the sample to ambient RH. Alternatives are a wet sponge plus desiccant chamber, desiccant alone, or heated sensors.

**Equations as printed** (📜 [0099]–[0123]):
- Haldane, eq. 1–4: $V_I = V_E\,\dfrac{0.99063-(F_EO_2+F_ECO_2)}{0.7808}$
- $VO_2 = V_I F_IO_2 - V_E F_EO_2$ and $VCO_2 = V_E F_ECO_2 - V_I F_ICO_2$, with **$F_IO_2 = 0.2005$** and **$F_ICO_2 = 0.00039$** used as constants.
- Weir: $REE = 1.44\,[3.9\,VO_2 + 1.1\,VCO_2]$ (kcal/day).
- RQ = VCO₂/VO₂. Read as fat ≈ 0.7, protein ≈ 0.8, carbohydrate ≈ 1.0.
- Waveform fit, eq. 5: $f_{O_2}(t) = -f_{CO_2}(t)\,c_1 + c_2$, with c₁ and c₂ positive.

**Validation** (📜 [0141], FIGs. 4–6):
- 25 healthy subjects were measured at the same time on the device and on an unnamed "gold-standard" device. The figures are Bland–Altman plots of VCO₂ (FIG. 4), energy expenditure (FIG. 5) and VO₂ (FIG. 6).
- The text says only that the device "has medical grade accuracy according to the norms of the industry". **It gives no numbers.**
- The plots have **no axis labels or units**. 💭 Read off the rendered figures, approximately (units presumably mL/min and kcal/day):

| Figure | Mean range (x) | Bias (dashed) | Upper LoA | Lower LoA |
|---|---|---|---|---|
| FIG. 4 VCO₂ | ~150–680 | ≈ −7 | ≈ +18 | ≈ −31 |
| FIG. 5 EE | ~1,200–3,200 | ≈ −48 | ≈ +100 | ≈ −200 |
| FIG. 6 VO₂ | ~190–480 | ≈ −7 | ≈ +16 | ≈ −30 |

**FIG. 3** (💭 my observation): the O₂ axis runs 0.14–0.22 and the CO₂ axis is labelled "ppm" with plateaus near 5,000. An O₂ drop of about 4.5 points would normally pair with about 4–5 % CO₂ (≈ 40,000–50,000 ppm). The time axis runs 0–6,000 "seconds" ([0040]) but shows about 6 breaths. Both axes are probably mislabelled, perhaps samples or ms and CO₂ ×10. Don't take FIG. 3 units literally.

**Two inconsistencies in the printed constants** (💭, checked numerically in section 6):
1. 0.99063 = 0.7808 (N₂) + 0.2095 (O₂) + ~0.0004 (CO₂). It is the dry-air sum without argon, so eq. 4 makes $V_I = V_E$ at zero gas exchange only if $F_IO_2 \approx 0.2095$.
2. The VO₂ formula then uses $F_IO_2 = 0.2005$, about the O₂ fraction of air carrying ~4.3 % water vapour. Mixing the two conventions biases VO₂ by about −20 % in our simulation. The fix, which the patent also hints at in [0072], is to **measure F_IO₂ on ambient air at the same sensor and the same humidity** (auto-zero, section 10).

**Granted claim 1 of US 12,736,522** (via Justia) adds all of these to the published claim:
- a humidity sensor and a temperature sensor
- O₂ and CO₂ sensors with **millisecond response rates**, in-line in the secondary pathway, measuring both inhaled and exhaled breath
- flow, O₂ and CO₂ sensors **heated** against condensation
- humidity and temperature compensation in the microcontroller
- a flow sensor that is a **hot-film anemometer or micro-TCD**
- a device that is wearable or handheld
- secondary tubes narrower than the primary pathway that regulate the flow
- and the device **"does not include a pump, a valve system, or a dehumidifying means"**.

Dependent claims include eq. 5 (claim 6), eq. 4 (claim 7), and a fluorescence O₂ sensor with a TCD CO₂ sensor (claim 8).

💭 So the granted independent claim describes a **passive (pumpless) sidestream**. A pumped design with a Nafion dryer (FIG. 1) is the embodiment the granted claim 1 excludes. That does **not** make such a design free to use: there may be other family members (EP 17782964.5, JP, BR, and WO2017180606, which is ceased) and other PNOE patents (e.g. EP3955813, "Respiratory analysis device"), and the doctrine of equivalents applies. **Get an FTO opinion before any commercial use.**
""")

md(r"""
## 1. Setup and the simulation core
Pure numpy/scipy/matplotlib. The core is five small blocks:
- constants and ground-truth generator
- sidestream and sensors
- device-side processing
- helpers
- plotting

Every numeric default carries a label.
""")
code(r"""
import numpy as np, math, time
import matplotlib.pyplot as plt
from scipy.signal import lfilter
np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
T_START = time.time()
""")
for i in range(len(CORE)):
    code(CORE[i])

md(r"""
**Textbook sources for the 📚 constants and formulas:**
- dry-air composition: O₂ 20.95 %, N₂ 78.08 %, Ar ≈ 0.93 %, standard atmosphere tables; CO₂ ≈ 420 ppm (NOAA global mean, early 2020s)
- water saturation pressure: Magnus form (Alduchov & Eskridge 1996); it gives 62.7 hPa = 47 mmHg at 37 °C, the classic BTPS value
- STPD = 0 °C, 1013.25 hPa, dry; BTPS = 37 °C, ambient pressure, saturated; ATPS = ambient temperature and pressure, saturated (or measured RH)
- Haldane transform: inert gas (N₂ + Ar) is neither consumed nor produced, so $V_I F_{IN_2} = V_E F_{EN_2}$ (e.g. ICALIC position paper, Oshima et al., *Clin Nutr*, in press 2016)
""")

md(r"""
## 2. Ground truth: synthetic breathing (🧪 every number here is an assumption)
- **Protocol:** 5 min seated rest, then a 25 W/min cycling ramp for 12 min.
- **VO₂:** 0.30 L/min at rest. During exercise it follows 0.55 + 10 mL/min/W with a 40 s first-order lag.
- **VCO₂:** RER 0.82 at rest. The VCO₂-vs-VO₂ slope is 0.95 below **VT1 (VO₂ = 1.6 L/min)** and 1.30 above it.
- **VE:** VE/VCO₂ = 28, rising above **VT2 (VO₂ = 2.8 L/min)**.
- **Breathing rate:** 12 → ~50 /min.
- **Variability:** tidal volume CV 8 %, breath duration CV 6 %.
- **Breath waveform:** half-sine flow. The capnogram shape (phase I dead space 150 mL, phase II rise, phase III slope) multiplies both ΔO₂ and ΔCO₂. The per-breath ΔO₂, ΔCO₂ and V_I are solved exactly, so each breath carries its true VO₂·T and VCO₂·T **and** N₂ balances (Haldane holds by construction).
""")
code(r"""
truth = Truth()
sig, br = gen_breaths(truth.T, truth.state, seed=0)
print(f"{len(br['k'])} breaths, {len(sig['t'])/FS:.0f} s at {FS:.0f} Hz")
fig, ax = plt.subplots(2, 2, figsize=(13, 6))
tg = truth.tg[truth.tg <= truth.T]; s = truth.state(tg)
ax[0,0].plot(tg/60, s["vo2"], label="VO2"); ax[0,0].plot(tg/60, s["vco2"], label="VCO2"); ax[0,0].set_ylabel("L/min STPD"); ax[0,0].legend(); ax[0,0].set_title("ground truth gas exchange")
ax[0,1].plot(tg/60, s["ve"], "g"); ax[0,1].set_ylabel("VE L/min BTPS"); a2 = ax[0,1].twinx(); a2.plot(tg/60, s["f"], "k--"); a2.set_ylabel("breaths/min"); ax[0,1].set_title("ventilation and rate")
w = (sig["t"] > 100) & (sig["t"] < 125)
ax[1,0].plot(sig["t"][w], sig["flow"][w]); ax[1,0].set_ylabel("flow L/s STPD (+exp)"); ax[1,0].set_title("rest: flow at the mouth")
ax2 = ax[1,1]; ax2.plot(sig["t"][w], sig["fo2"][w]*100, "b"); ax2.set_ylabel("O2 % (dry)", color="b"); a3 = ax2.twinx(); a3.plot(sig["t"][w], sig["fco2"][w]*100, "r"); a3.set_ylabel("CO2 % (dry)", color="r"); ax2.set_title("mouth gas: mirror images (patent eq. 5 idea)")
for a in ax[1]: a.set_xlabel("s")
plt.tight_layout(); plt.show()
print("true breath VI/VE (rest):", np.round((br["VIb"]/br["VEb"])[:5], 4), "- not 1, because RER < 1 (Haldane)")
""")

md(r"""
## 3. The sidestream: transport delay, mixing, first-order sensors
- **Transport delay** = tube volume / pump flow = $\pi (ID/2)^2 L / Q_s$ (📚 plug flow).
- **Mixing:** gas also smears as it travels (Taylor dispersion). 🧪 We model the smear as an extra first-order lag of 10 % of the transport delay.
- **Sensors:** each is a first-order system, $\tau\,\dot y = x - y$ (🧪 a standard simplification).
- **Order:** the CO₂ sensor sits upstream of the O₂ sensor (📜 either order is allowed, [0059]), so O₂ sees the gas another $V_{between}/Q_s$ later.
- **Defaults** (🧪, not from the patent): 1.0 m × 1.2 mm ID tube, 150 mL/min pump, τ_CO₂ = 0.10 s, τ_O₂ = 0.15 s, 0.3 mL between the sensors, Nafion dryer. Noise: O₂ 0.02 % and CO₂ 0.01 % absolute, flow 1 %.
""")
code(r"""
hw = Hardware()
env = Env()
m = measure(sig, hw, env)
print(f"tube volume {hw.v_tube_ml:.2f} mL -> transport delay {hw.transport_delay*1000:.0f} ms; O2 sees gas {hw.between_delay*1000:.0f} ms after CO2")
w = (sig["t"] > 100) & (sig["t"] < 115)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
ax[0].plot(sig["t"][w], sig["fco2"][w]*100, "k", label="CO2 at mouth (dry)")
ax[0].plot(m["t"][w], m["co2"][w]*100, "r", label="CO2 sensor (wet, delayed, filtered)")
ax[0].legend(); ax[0].set_ylabel("%"); ax[0].set_title("what the CO2 sensor sees")
ax[1].plot(sig["t"][w], sig["fo2"][w]*100, "k", label="O2 at mouth (dry)")
ax[1].plot(m["t"][w], m["o2"][w]*100, "b", label="O2 sensor")
ax[1].legend(); ax[1].set_title("O2: later and slower than CO2")
for a in ax: a.set_xlabel("s")
plt.tight_layout(); plt.show()
""")

md(r"""
## 4. Humidity and BTPS / ATPS / STPD
Three separate humidity effects:
1. **Flow volume.** The flow sensor sees exhaled gas warm and saturated (🧪 34 °C in the mask) and inhaled gas at room conditions (22 °C, 40 % RH). Both must go to STPD-dry before Haldane. Treating exhaled gas as room air (ATPS) is a classic error.
2. **Gas dilution.** Water vapour dilutes O₂ and CO₂: $F_{wet} = F_{dry}(1-x_{H_2O})$. Exhaled gas carries about 6 % water at 37 °C against about 1 % in room air. Without drying, the expired O₂ reading drops by about 5 % relative. That looks like **huge** extra O₂ uptake unless it is corrected.
3. **📜 The patent's answer** is Nafion (sample → ambient RH) plus an RH/T/P sensor for compensation ([0024], [0063]–[0065], [0085]). The granted claim 1 instead uses **heated sensors, no dryer**, and humidity compensation.
""")
code(r"""
pa = env.pH2O_amb
for name, T, ph in [("BTPS (37C, sat)", 37.0, p_sat(37.0)), ("mask exhalate (34C, sat)", 34.0, p_sat(34.0)), ("ATPS room (22C, 40%RH)", env.T_amb, pa), ("ATPS room saturated", env.T_amb, p_sat(env.T_amb))]:
    print(f"{name:28s} -> STPD factor {to_stpd_factor(env.PB, T, ph):.4f}   water fraction {ph/env.PB*100:.2f} %")
print("BTPS->STPD / ATPS->STPD ratio:", round(to_stpd_factor(env.PB, 34, p_sat(34))/to_stpd_factor(env.PB, 22, pa), 4))

B = detect_breaths(m["flow"])
d = estimate_gas_delay(m, B, hw.tau_co2); f5 = fit_o2_vs_co2({"o2": m["o2"][:6000], "co2": m["co2"][:6000]})
ALIGN = dict(gas_delay=d["align"], o2_extra_lag=f5["lag_s"] + f5["extra_tau_s"])
rest_mask = lambda r: r["t"] < truth.rest_s
def summarize(label, r):
    tv, tc = match_truth(r, sig, br); e = (r["VO2"] - tv) / tv * 100; ec = (r["VCO2"] - tc) / tc * 100
    R = rest_mask(r); R[:3] = False; X = ~rest_mask(r)
    print(f"{label:52s} VO2 err rest {e[R].mean():+6.1f}% (sd {e[R].std():4.1f}) | ramp {e[X].mean():+6.1f}% | VCO2 rest {ec[R].mean():+6.1f}%")
    return e
summarize("Nafion, correct BTPS handling", bbb(m, B, env, **ALIGN))
summarize("Nafion, exhaled flow treated as ATPS (mistake)", bbb(m, B, env, exp_as="atps", **ALIGN))
m_wet = measure(sig, Hardware(dryer="none"), env)
summarize("no dryer, device assumes ambient humidity", bbb(m_wet, B, env, **ALIGN))
summarize("no dryer, compensated by slow in-line RH sensor", bbb(m_wet, B, env, use_rh=True, **ALIGN))
""")
md(r"""
🧪 With no dryer and a 1 s humidity sensor, the RH reading cannot follow breath-by-breath swings, so compensation is only partial. That is why a dryer, or a fast humidity estimate, matters. The granted claim's heated-sensor route needs one of the two.
""")

md(r"""
## 5. Recovering the time offsets
The flow signal is instantaneous, but the gas signals arrive later. A wrong delay mis-pairs gas with flow and corrupts VO₂ (see section 12).
- **Flow↔CO₂ offset** (🧪 our method): at each switch from expiration to inspiration, fresh air replaces alveolar gas *instantly*, so the CO₂ trace falls sharply. Take the median time from flow reversal to the 50 % CO₂ fall, then add τ(1−ln 2) to align centres of mass (a first-order sensor's 50 % point lags τ·ln 2, its centroid lags τ). This needs the delay to be shorter than one breath period, so calibrate it at rest or with a puff test.
- **O₂↔CO₂ offset** (📜 eq. 5, $f_{O_2} = -c_1 f_{CO_2} + c_2$): because the two waveforms are mirror images, a least-squares fit over lag (💭 plus an extra first-order lag for the slower O₂ sensor) recovers the O₂ sensor's extra delay. The patent doesn't say eq. 5 is used for timing. 💭 This use is our reading: a fit is only good when the two signals are aligned.
""")
code(r"""
true_eff = hw.transport_delay * (1 + hw.mix_frac) + hw.tau_co2
print(f"flow->CO2: t50 = {d['t50']*1000:.0f} ms, alignment delay {d['align']*1000:.0f} ms (simulated truth {true_eff*1000:.0f} ms)")
print(f"eq.5 fit: R^2={f5['r2']:.5f}  c1={f5['c1']:.3f}  c2={f5['c2']:.4f}  lag={f5['lag_s']*1000:.0f} ms + extra tau {f5['extra_tau_s']*1000:.0f} ms"
      f"  (truth: {hw.between_delay*1000:.0f} ms between sensors + {(hw.tau_o2-hw.tau_co2)*1000:.0f} ms slower O2)")
lags = np.arange(0, 0.6, 0.01); r2s = []
for L in lags:
    i = int(round(L*FS)); x, y = m["co2"][:6000-i], m["o2"][i:6000]
    r2s.append(np.corrcoef(-x, y)[0, 1]**2)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].plot(lags*1000, r2s); ax[0].set_xlabel("O2 lag vs CO2 (ms)"); ax[0].set_ylabel("R^2 of eq.5 fit"); ax[0].set_title("eq. 5 fit quality vs lag")
i = int(round((f5["lag_s"])*FS)); xs = first_order(m["co2"], f5["extra_tau_s"])
ax[1].plot(m["t"][:1500], m["o2"][:1500]*100, label="O2 sensor"); ax[1].plot(m["t"][:1500], (-f5["c1"]*xs[:1500]+f5["c2"])*100, "--", label="-c1*CO2+c2 (shifted)")
ax[1].set_xlim(10, 15); ax[1].legend(); ax[1].set_title("mirrored CO2 overlays O2 once aligned")
plt.tight_layout(); plt.show()
""")

md(r"""
## 6. Breath detection, then breath-by-breath VO₂/VCO₂ via Haldane
- **Breath detection:** zero-crossings of flow with ±0.03 L/s hysteresis and a minimum 0.25 s phase (🧪).
- **Per breath:**
  - $V_E = \int_{exp} \dot V\,dt$ (STPD)
  - $F_EO_2 = \int \dot V F_{O_2} / V_E$ (flow-weighted mixed-expired fraction, from the aligned and dried signal)
  - $V_I$ by Haldane
  - VO₂ = (V_I F_IO₂ − V_E F_EO₂)/T_breath (📚 / 📜 same algebra)

The three Haldane variants below are run on **ideal** sensors, so only the constants differ.
""")
code(r"""
ideal = Hardware(tube_len_m=0, tau_co2=0, tau_o2=0, mix_frac=0, v_between_ml=0, noise_o2=0, noise_co2=0, noise_flow=0)
mi = measure(sig, ideal, env); Bi = detect_breaths(mi["flow"])
print("detected", len(Bi), "breaths vs", len(br["k"]), "true")
summarize("ideal sensors, textbook Haldane (F_IN2 = 1-F_IO2-F_ICO2)", bbb(mi, Bi, env))
summarize("ideal, patent eq.4 (0.99063/0.7808) with F_IO2=0.2095", bbb(mi, Bi, env, haldane="patent"))
summarize("ideal, patent eq.4 + patent F_IO2=0.2005, F_ICO2=0.00039", bbb(mi, Bi, env, haldane="patent", FiO2=PAT_FIO2, FiCO2=PAT_FICO2))
r_main = bbb(m, B, env, **ALIGN)
e_main = summarize("realistic default hardware, delays recovered (sec. 5)", r_main)
tv, tc = match_truth(r_main, sig, br)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
ax[0].plot(r_main["t"]/60, r_main["VO2"], ".", ms=3, label="device VO2"); ax[0].plot(r_main["t"]/60, tv, "k-", lw=1, label="truth")
ax[0].plot(r_main["t"]/60, r_main["VCO2"], ".", ms=3, label="device VCO2"); ax[0].plot(r_main["t"]/60, tc, "k--", lw=1)
ax[0].set_xlabel("min"); ax[0].set_ylabel("L/min"); ax[0].legend(); ax[0].set_title("breath-by-breath, realistic hardware")
ax[1].plot(r_main["t"]/60, e_main, ".", ms=3); ax[1].set_xlabel("min"); ax[1].set_ylabel("VO2 error %"); ax[1].set_title("error grows as breaths get shorter (sec. 8)")
plt.tight_layout(); plt.show()
""")
md(r"""
💭 **Takeaways:**
- The patent's Haldane form is fine with a dry-air F_IO₂.
- Combined with its printed F_IO₂ = 0.2005, it under-reads VO₂ by about a fifth in this simulation.
- With realistic sensors (τ ≈ 0.1–0.15 s), resting breaths come out accurate. Fast exercise breaths (up to ~50 /min) under-read, because τ is no longer small compared with a ~0.6 s exhalation.
""")

md(r"""
## 7. RER, Weir REE and Frayn substrate oxidation (📚)
- **Weir** (J. Physiol. 1949;109:1–9): kcal = 3.941·L O₂ + 1.106·L CO₂ − 2.17·g urinary N. Weir's abbreviated form is 3.9·O₂ + 1.1·CO₂.
  - With VO₂ in **mL/min**, REE (kcal/day) = 1.44·(3.941·VO₂ + 1.106·VCO₂), because 1.44 = 1440 min/day ÷ 1000 mL/L.
  - 📜 The patent prints the abbreviated 1.44[3.9·VO₂ + 1.1·VCO₂].
- **Frayn** (J. Appl. Physiol. 1983;55:628–634), in g/min with VO₂ and VCO₂ in L/min and n = urinary N in g/min:
  - carbohydrate = 4.55·VCO₂ − 3.21·VO₂ − 2.87·n
  - fat = 1.67·VO₂ − 1.67·VCO₂ − 1.92·n
- **Validity:** these hold only at steady state with no net lactate or bicarbonate CO₂ (Frayn discusses the lactate and ketone disturbances). They don't hold above VT1/VT2 on a ramp, where buffering adds non-metabolic CO₂. 🧪 We use n = 0.
- 📜 The patent's RQ reading (0.7 fat, 0.8 protein, 1.0 carbohydrate) is a rough guide. The textbook non-protein RQ for pure fat is about 0.70.
""")
code(r"""
R = rest_mask(r_main); R[:3] = False
vo2, vco2 = r_main["VO2"][R].mean(), r_main["VCO2"][R].mean()
tvo2, tvco2 = tv[R].mean(), tc[R].mean()
weir = lambda o, c: 1.44 * (3.941 * o * 1000 + 1.106 * c * 1000)
weir_pat = lambda o, c: 1.44 * (3.9 * o * 1000 + 1.1 * c * 1000)
print(f"rest: device VO2 {vo2*1000:.0f} mL/min, VCO2 {vco2*1000:.0f}, RER {vco2/vo2:.3f}   | truth {tvo2*1000:.0f}, {tvco2*1000:.0f}, RER {tvco2/tvo2:.3f}")
print(f"REE Weir (3.941/1.106): device {weir(vo2, vco2):.0f} kcal/day, truth {weir(tvo2, tvco2):.0f};  patent form (3.9/1.1): {weir_pat(vo2, vco2):.0f}")
cho = lambda o, c, n=0: 4.55*c - 3.21*o - 2.87*n
fat = lambda o, c, n=0: 1.67*o - 1.67*c - 1.92*n
print(f"Frayn at rest: CHO {cho(vo2, vco2)*1000:.0f} mg/min, fat {fat(vo2, vco2)*1000:.0f} mg/min (truth {cho(tvo2, tvco2)*1000:.0f}, {fat(tvo2, tvco2)*1000:.0f})")
tb, (bo, bc, to_, tc_) = bin_avg(r_main["t"], [r_main["VO2"], r_main["VCO2"], tv, tc], 20)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
ax[0].plot(tb/60, bc/bo, "o-", ms=3, label="device RER (20 s bins)"); ax[0].plot(tb/60, tc_/to_, "k--", label="truth"); ax[0].axhline(1, color="gray", lw=0.8); ax[0].legend(); ax[0].set_xlabel("min"); ax[0].set_title("RER")
ok = to_ < truth.vt1
ax[1].plot(tb[ok]/60, cho(bo, bc)[ok]*60, label="CHO g/h"); ax[1].plot(tb[ok]/60, fat(bo, bc)[ok]*60, label="fat g/h"); ax[1].legend(); ax[1].set_xlabel("min"); ax[1].set_title("Frayn (shown only below VT1)")
plt.tight_layout(); plt.show()
""")

md(r"""
## 8. Slow sensors: breath skipping, sampling chamber, fast sensors, software speed-up
📜 The patent says that at about 40–50 breaths/min the response time of the sensors may not be enough, so only every 2nd, 3rd or 4th breath is analysed ([0077]). It doesn't say what happens downstream during a skipped breath.

First, the physics. We hold the state steady at a given breathing rate, give both sensors the same τ, and align on the centroid, then measure the VO₂ bias.
""")
code(r"""
def steady(f_bpm, dur=90, seed=0):
    vs = np.linspace(0.3, 3.6, 400)
    vc = truth.vco2_of_vo2(vs); ve = truth.ve_of(vs, vc); ff = np.clip(12 + 36*(ve-8)/122, 12, 50)
    j = int(np.argmin(abs(ff - f_bpm))); v, c_, e_ = vs[j], float(vc[j]), float(ve[j])
    st = lambda t: dict(vo2=v, vco2=c_, ve=e_, f=f_bpm, ti_frac=0.40 + 0.08*(f_bpm-12)/36, W=0)
    return gen_breaths(dur, st, seed=seed)
base = dict(tube_len_m=0.0, mix_frac=0, noise_o2=0, noise_co2=0, noise_flow=0, v_between_ml=0)
taus = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]; rates = [12, 25, 35, 45]
tab = np.zeros((len(rates), len(taus)))
for i, fb in enumerate(rates):
    s_, b_ = steady(fb)
    for j, tau in enumerate(taus):
        mm = measure(s_, Hardware(**base, tau_co2=tau, tau_o2=tau), env); BB = detect_breaths(mm["flow"])
        tab[i, j] = err_stats(bbb(mm, BB, env, gas_delay=tau), s_, b_)["vo2_bias"]
print("VO2 bias % (rows: breaths/min, cols: sensor tau s), centroid-aligned, every breath analysed")
print("bpm\\tau " + "".join(f"{t:>8}" for t in taus))
for i, fb in enumerate(rates): print(f"{fb:>7} " + "".join(f"{x:8.1f}" for x in tab[i]))
""")
md(r"""
Now four ways to live with a **slow sensor (τ = 0.8 s) at 45 breaths/min**, compared with a **fast sensor (τ = 0.05 s)**:
- **(a)** analyse every breath anyway
- **(b)** 💭 the "skip + sampling chamber" reading of the patent: collect breath *k*'s sample in a small chamber (📜 claim 16 / [0093] allow one), close the valve, and let the slow sensor settle during the N−1 skipped breaths. Two pump options:
  - constant pump flow, which gives a **time-weighted** average
  - 🧪 pump flow proportional to expiratory flow, which gives a **flow-weighted** average, as Haldane needs
- **(c)** 🧪 software speed-up $x + \tau\dot x$ (inverse first-order filter). It works if τ is known and the noise is low.
""")
code(r"""
s45, b45 = steady(45, dur=120); B45 = detect_breaths(s45["flow"])
def hold_vo2(out):
    errs = []
    for bi, fo, fc in out:
        a, b, c = B45[bi]; q = s45["flow"][b:c].clip(min=0); VE = q.sum()*DT
        VI = VE*(1-fo-fc)/(1-FiO2_DRY-FiCO2_DRY); T = (c-a)*DT
        k = s45["bidx"][b+2]; tvv = b45["VO2b"][k]/b45["Ttot"][k]*60
        errs.append(((VI*FiO2_DRY - VE*fo)/T*60 - tvv)/tvv*100)
    e = np.array(errs[2:]); return e.mean(), e.std(), len(e)
rows = []
for tau, lab in [(0.05, "fast sensor tau=0.05 s, every breath"), (0.8, "slow tau=0.8 s, every breath")]:
    mm = measure(s45, Hardware(**dict(base, noise_o2=2e-4, noise_co2=1e-4), tau_co2=tau, tau_o2=tau), env)
    es = err_stats(bbb(mm, B45, env, gas_delay=tau), s45, b45); rows.append((lab, es["vo2_bias"], es["vo2_sd"], es["n"]))
mm = measure(s45, Hardware(**dict(base, noise_o2=2e-4, noise_co2=1e-4), tau_co2=0.8, tau_o2=0.8), env)
es = err_stats(bbb(mm, B45, env, gas_delay=0.0, speedup_tau=(0.8, 0.8)), s45, b45)
rows.append(("slow tau=0.8 s + software speed-up (noise 0.02%)", es["vo2_bias"], es["vo2_sd"], es["n"]))
for N in [2, 3, 4]:
    for prop in [False, True]:
        mu, sd, n = hold_vo2(sample_and_hold(s45, B45, N, 0.45, 0.8, proportional=prop))
        rows.append((f"slow, skip N={N} + chamber, {'flow-proportional' if prop else 'constant'} pump", mu, sd, n))
print(f"{'strategy at 45 breaths/min':58s} {'bias%':>7} {'sd%':>6} {'breaths':>8}")
for lab, mu, sd, n in rows: print(f"{lab:58s} {mu:7.1f} {sd:6.1f} {n:8d}")
""")
md(r"""
💭 **Reading the table** (simulation only):
- Skipping on its own doesn't fix a slow sensor. What helps is giving it **time on a held sample**, and the sample must be **flow-weighted**, or the result is a time-weighted mean that over-weights the end of the breath (alveolar gas at low flow).
- Analysing every 3rd–4th breath means 1/3–1/4 of the breaths go unmeasured. For VT detection on 10–20 s bins that is usually acceptable.
- Fast sensors, or a well-characterised software speed-up, keep every breath.
""")

md(r"""
## 9. Mask leaks: why the circuit must be sealed
- An **expiratory leak** (gas escaping round the seal before the flow sensor) makes V_E under-read. With Haldane, VO₂ ∝ V_E, so the error ≈ −leak %.
- An **inspiratory leak** doesn't matter for an expiration-only Haldane device. It **wrecks** a device that integrates inspired flow directly, because VO₂ is a ~4 % difference of two large volumes.
- 📜 The patent offers a one-way valve option ([0082]–[0084]). The granted claim 11 uses a facemask *without* a non-rebreathing valve.
""")
code(r"""
print(f"{'case':55s} {'VO2 rest':>9} {'VO2 ramp':>9}")
for lab, kw, meth in [("sealed, Haldane", {}, "haldane"), ("exp leak 3%, Haldane", dict(leak_exp=0.03), "haldane"),
                      ("exp leak 10%, Haldane", dict(leak_exp=0.10), "haldane"),
                      ("exp leak scaling with flow (5% at peak exercise flow)", dict(leak_exp=0.05, leak_mode="flow"), "haldane"),
                      ("insp leak 5%, Haldane (expiration only)", dict(leak_insp=0.05), "haldane"),
                      ("sealed, direct VI integration", {}, "direct"), ("insp leak 2%, direct VI integration", dict(leak_insp=0.02), "direct")]:
    mm = measure(sig, Hardware(**kw), env); BB = detect_breaths(mm["flow"])
    r = bbb(mm, BB, env, method=meth, **ALIGN); tvv, _ = match_truth(r, sig, br); e = (r["VO2"] - tvv)/tvv*100
    Rm = rest_mask(r); Rm[:3] = False
    print(f"{lab:55s} {e[Rm].mean():+8.1f}% {e[~rest_mask(r)].mean():+8.1f}%")
""")

md(r"""
## 10. Ambient auto-zero between breaths
📜 The pump draws ambient air in the dead time between exhalations, "useful for calibration" ([0072]). 🧪 We simulate the drift that makes this necessary: O₂ sensor gain drifting +2 % and CO₂ offset +0.05 % over the 17-minute test. Each breath's late-inspiration window (ambient air after alignment) then re-scales O₂ to 20.95 % and re-zeros CO₂.
""")
code(r"""
md_ = measure(sig, Hardware(o2_gain_drift=0.02, co2_offset_drift=0.0005), env); Bd = detect_breaths(md_["flow"])
outs = {}
for lab, src, BB, kw in [("no drift, constant F_IO2=0.2095 (reference)", m, B, {}),
                         ("drift, constant F_IO2=0.2095 (no auto-zero)", md_, Bd, {}),
                         ("drift, auto-zero on ambient each breath", md_, Bd, dict(autozero=True)),
                         ("no drift, patent constant F_IO2=0.2005", m, B, dict(FiO2=PAT_FIO2, FiCO2=PAT_FICO2))]:
    r = bbb(src, BB, env, **kw, **ALIGN); tvv, _ = match_truth(r, sig, br); outs[lab] = (r["t"], (r["VO2"]-tvv)/tvv*100)
    Rm = rest_mask(r); Rm[:3] = False
    print(f"{lab:46s} VO2 error rest {outs[lab][1][Rm].mean():+6.1f}% | last 2 min {outs[lab][1][r['t'] > truth.T-120].mean():+6.1f}%")
plt.figure(figsize=(10, 3.4))
for lab, (t_, e_) in outs.items(): plt.plot(t_/60, e_, ".", ms=2, label=lab)
plt.ylim(-40, 15); plt.xlabel("min"); plt.ylabel("VO2 error %"); plt.legend(fontsize=8); plt.title("auto-zero removes slow drift"); plt.show()
""")

md(r"""
## 11. VT1 / VT2 on the ramp (📚 V-slope and ventilatory equivalents)
- **V-slope** (Beaver, Wasserman & Whipp, *J Appl Physiol* 1986;60:2020–2027): VT1 is the breakpoint where the VCO₂-vs-VO₂ slope steepens.
- **Ventilatory equivalents:** VT1 is where VE/VO₂ starts rising while VE/VCO₂ stays flat. VT2 (respiratory compensation point) is where VE/VCO₂ rises.
- 🧪 Method: 10 s bins on the ramp, then a two-segment least-squares breakpoint search. VT2 is found first (VE vs VCO₂); V-slope then uses only data below VT2. The simulated truth is VT1 at VO₂ 1.6 and VT2 at VO₂ 2.8 L/min.
""")
code(r"""
X = r_main["t"] > truth.rest_s + 90
tb, (o, c, ve) = bin_avg(r_main["t"][X], [r_main["VO2"][X], r_main["VCO2"][X], r_main["VE"][X]], 10)
ve_btps = ve / to_stpd_factor(env.PB, 37.0, p_sat(37.0))
vt2_vco2, _ = two_segment(c, ve_btps); vt2_vo2 = np.interp(vt2_vco2, np.sort(c), o[np.argsort(c)])
below = o < vt2_vo2
vt1_vo2, coef = two_segment(o[below], c[below])
eq_o2, eq_co2 = ve_btps/o, ve_btps/c
vt1_eq, _ = two_segment(o[below], eq_o2[below]); vt2_eq, _ = two_segment(o, eq_co2, 0.3, 0.95)
print(f"VT1: V-slope {vt1_vo2:.2f} L/min, VE/VO2 breakpoint {vt1_eq:.2f}   (truth {truth.vt1})")
print(f"VT2: VE-vs-VCO2 {vt2_vo2:.2f} L/min, VE/VCO2 breakpoint {vt2_eq:.2f}  (truth {truth.vt2})")
# the same thresholds expressed as time on the ramp (device VO2 under-reads at high breathing rates, truth does not)
t_dev = lambda v: tb[np.argmax(o >= v)]
tt = truth.tg; tvo2_curve = truth.vo2_g
t_true = lambda v: tt[np.argmax(tvo2_curve >= v)]
print(f"as ramp time: VT1 device {t_dev(vt1_vo2)/60:.1f} min vs truth {t_true(truth.vt1)/60:.1f} min; VT2 device {t_dev(vt2_vo2)/60:.1f} min vs truth {t_true(truth.vt2)/60:.1f} min")
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(o, c, ".", label="10 s bins"); xx = np.linspace(o[below].min(), o[below].max(), 50)
ax[0].plot(xx, coef[0] + coef[1]*xx + coef[2]*np.clip(xx - vt1_vo2, 0, None), "r-", label="two-segment fit")
ax[0].axvline(vt1_vo2, color="r", ls=":"); ax[0].axvline(truth.vt1, color="k", ls="--", lw=0.8, label="truth VT1")
ax[0].set_xlabel("VO2 L/min"); ax[0].set_ylabel("VCO2 L/min"); ax[0].legend(); ax[0].set_title("V-slope")
ax[1].plot(o, eq_o2, "b.", label="VE/VO2"); ax[1].plot(o, eq_co2, "r.", label="VE/VCO2")
for v, cc in [(vt1_eq, "b"), (vt2_eq, "r")]: ax[1].axvline(v, color=cc, ls=":")
for v in (truth.vt1, truth.vt2): ax[1].axvline(v, color="k", ls="--", lw=0.8)
ax[1].set_xlabel("VO2 L/min"); ax[1].legend(); ax[1].set_title("ventilatory equivalents (BTPS VE)")
plt.tight_layout(); plt.show()
""")

md(r"""
## 12. Design knobs ✏️ (edit and re-run)
These tie the hardware to the breath-by-breath accuracy you can expect:
- **delay** = π(ID/2)²·L / Q_pump
- **smear**, set by the mixing fraction and the sensor τ
- **sensitivity to delay error**: a wrong alignment mis-pairs gas and flow

🧪 For PRANA-style layouts (face piece + body pod), a long tube raises the delay. That is harmless if it is measured and corrected, and harmful if it drifts (pump-flow changes, a kinked tube).
""")
code(r"""
# ---- ✏️ your design ----
DESIGN = dict(tube_len_m=1.0, tube_id_mm=1.2, pump_ml_min=150.0, tau_co2=0.10, tau_o2=0.15, v_between_ml=0.3)
F_BPM = 45            # breathing rate to test at
DELAY_ERROR_MS = 0    # mis-estimate of the alignment delay
# ------------------------
def design_report(design, f_bpm, delay_err_ms=0, seed=3):
    hwd = Hardware(**design); s_, b_ = steady(f_bpm, dur=60, seed=seed)
    mm = measure(s_, hwd, env); BB = detect_breaths(mm["flow"])
    dd = estimate_gas_delay(mm, BB, hwd.tau_co2); ff = fit_o2_vs_co2({"o2": mm["o2"][:3000], "co2": mm["co2"][:3000]}, max_lag_s=0.6)
    es = err_stats(bbb(mm, BB, env, gas_delay=dd["align"] + delay_err_ms/1000, o2_extra_lag=ff["lag_s"] + ff["extra_tau_s"]), s_, b_)
    return hwd, dd, es
hwd, dd, es = design_report(DESIGN, F_BPM, DELAY_ERROR_MS)
print(f"tube {hwd.v_tube_ml:.2f} mL -> transport delay {hwd.transport_delay*1000:.0f} ms (+ mixing {hwd.mix_frac*hwd.transport_delay*1000:.0f} ms); recovered alignment {dd['align']*1000:.0f} ms")
print(f"at {F_BPM} breaths/min: VO2 bias {es['vo2_bias']:+.1f}% sd {es['vo2_sd']:.1f}% | VCO2 bias {es['vco2_bias']:+.1f}% sd {es['vco2_sd']:.1f}%")

# sweeps
errs_ms = [-100, -50, -20, 0, 20, 50, 100]
sweep = [design_report(DESIGN, F_BPM, e)[2]["vo2_bias"] for e in errs_ms]
tau_grid = [0.03, 0.06, 0.1, 0.15, 0.25, 0.4]; pumps = [50, 100, 150, 250]
heat = np.array([[design_report(dict(DESIGN, pump_ml_min=p, tau_o2=t, tau_co2=min(t, 0.1)), F_BPM)[2]["vo2_bias"] for t in tau_grid] for p in pumps])
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
ax[0].plot(errs_ms, sweep, "o-"); ax[0].set_xlabel("alignment delay error (ms)"); ax[0].set_ylabel("VO2 bias %"); ax[0].set_title(f"delay error at {F_BPM}/min")
im = ax[1].imshow(heat, cmap="RdBu", vmin=-30, vmax=30, aspect="auto")
ax[1].set_xticks(range(len(tau_grid))); ax[1].set_xticklabels(tau_grid); ax[1].set_yticks(range(len(pumps))); ax[1].set_yticklabels(pumps)
ax[1].set_xlabel("O2 sensor tau (s)"); ax[1].set_ylabel("pump mL/min"); ax[1].set_title(f"VO2 bias % at {F_BPM}/min")
for i in range(len(pumps)):
    for j in range(len(tau_grid)): ax[1].text(j, i, f"{heat[i,j]:.0f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=ax[1]); plt.tight_layout(); plt.show()
""")
md(r"""
💭 Pump flow mostly changes the delay (and the mixing smear, which grows with it), and alignment corrects the delay. Sensor τ is the knob that limits breath-by-breath accuracy at high breathing rates. Delay *errors* of a few tens of ms already cost several % at 45 /min.
""")

md(r"""
## 13. Summary

| Item | Status |
|---|---|
| Primary/secondary pathways, flow sensor in primary, O₂+CO₂ **in-line** in secondary | 📜 [0012], claim 1 |
| Nafion dryer, pump, 3-way valve, breath skipping at ~40–50 /min | 📜 [0065], [0072], [0077] (excluded from granted claim 1) |
| Ambient air drawn between breaths for calibration | 📜 [0072] |
| Haldane with 0.99063 / 0.7808; F_IO₂ = 0.2005, F_ICO₂ = 0.00039 | 📜 [0099]–[0121]; 💭 internally inconsistent (sec. 6) |
| Weir 1.44[3.9 VO₂ + 1.1 VCO₂] | 📜 [0113]; 📚 Weir 1949 (3.941 / 1.106 full form) |
| eq. 5 O₂ = −c₁·CO₂ + c₂ | 📜 [0112], granted claim 6; 💭 used here for timing |
| Validation: 25 healthy subjects, Bland–Altman, no numbers in text | 📜 [0141]; 💭 approximate LoA read off FIGs. 4–6 |
| Frayn equations, BTPS/STPD, V-slope | 📚 Frayn 1983; standard gas laws; Beaver et al. 1986 |
| All physiology, hardware defaults, noise, leak and drift values | 🧪 |
| Chamber + flow-proportional pump as a reading of "skip" mode | 💭 / 🧪 |

**Caveats:**
- This is a simulator. It shows *mechanisms* and *relative* sensitivities.
- Real sensors aren't first-order. Water and temperature affect them in more complex ways, flow sensors have their own dynamics, and real masks have dead space that adds rebreathing.
- Validate any build against a reference system (e.g. a metabolic simulator, or a calibrated syringe plus gas mixes) before trusting it.
- **Not medical advice. Not legal advice. Get a freedom-to-operate review before any commercial use.**
""")
code(r"""
print(f"notebook finished in {time.time() - T_START:.0f} s")
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}}
nbf.validate(nb)
nbf.write(nb, os.path.join(HERE, "notebook.ipynb"))
print("cells:", len(cells))
