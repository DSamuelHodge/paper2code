# Breath Analysis Device (US 2019/0120821 A1): Paper2Code, patent edition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DSamuelHodge/paper2code/blob/main/papers/physiological-sensing/patent-US20190120821A1-sidestream-vo2/notebook.ipynb)

**Source:** US 2019/0120821 A1, *Breath Analysis Device*. Inventor Apostolos Atsalakis; applicant Endo Medical Inc. Published 25 Apr 2019. Appl. 16/092,988 (PCT/US17/26979, filed 11 Apr 2017); priority GB 1606292.9 and EP 17386006.5. [Google Patents](https://patents.google.com/patent/US20190120821A1/en) · [PDF](https://patentimages.storage.googleapis.com/5c/e8/03/4010b9c8ca2d6f/US20190120821A1.pdf)

**Status (checked 23 Nov 2026):** granted as **US 12,736,522 B2 on 15 Sep 2026**. The assignee is **PNOE Inc.**: Endo Medical's name change was recorded 5 Jan 2023. The prosecution history shows several final rejections and an appeal before the issue fee was received on 7 Jul 2026. Sources: [Justia](https://patents.justia.com/patent/12736522) and the Google Patents legal events.

**Granted claim 1** is narrower than the published one. It requires all of the following:
- millisecond-response O₂ and CO₂ sensors, in-line in the sidestream, measuring inhaled and exhaled breath
- humidity and temperature sensors, with compensation
- heated flow, O₂ and CO₂ sensors
- a hot-film or micro-TCD flow sensor
- a device that is wearable or handheld
- narrow tubes that regulate the sidestream flow
- **no pump, no valve system, no dehumidifier**

**Official code:** none, since this is a patent. The notebook is a from-scratch numpy/scipy simulator. Patent statements are tagged 📜, textbook facts 📚, simulation assumptions 🧪 and my interpretations 💭.

> ⚠️ **Not legal advice.** A personal, non-commercial build is a different situation from selling. Before you make, sell or offer a device or kit based on this architecture, get a freedom-to-operate review from a patent attorney. It should cover US 12,736,522, its family (EP 17782964.5, JP, BR) and PNOE's other patents.

## What's here
| File | What it is |
|---|---|
| `notebook.ipynb` | The Colab notebook: pure numpy/scipy/matplotlib, CPU, about 6 s. |
| `notebook_smoke_executed.ipynb` | A CPU run with all outputs. |
| `vo2sim_core.py` | The simulation core. `build_notebook.py` embeds it into the notebook, so the notebook stays self-contained. |
| `build_notebook.py` | Regenerates `notebook.ipynb`. |
| `run_smoke.py` | The smoke test (nbclient). |

## Notebook contents
0. What the patent says: architecture, equations, validation (with approximate Bland–Altman limits read off FIGs. 4–6), inconsistencies in the printed constants, and the granted claims.
1. Simulation core.
2. Synthetic breathing: rest, then a 25 W/min ramp, with an exact per-breath mass balance.
3. Sidestream transport delay, mixing, and first-order O₂/CO₂ sensors.
4. Humidity and BTPS/ATPS/STPD: dryer vs no dryer, and the "exhaled gas treated as ATPS" mistake.
5. Recovering time offsets: the flow↔CO₂ edge, and the patent's eq. 5 O₂ = −c₁·CO₂ + c₂ fit.
6. Breath detection and breath-by-breath Haldane VO₂/VCO₂, including the patent's own constants.
7. RER, Weir REE, Frayn substrate oxidation.
8. Slow sensors: an error table over τ and breathing rate; breath skipping with a sampling chamber (constant vs flow-proportional pump); software speed-up.
9. Mask leaks: Haldane vs direct inspired-flow integration.
10. Ambient auto-zero against sensor drift.
11. VT1/VT2 by V-slope and ventilatory equivalents.
12. Design knobs ✏️: tube length and ID, pump flow and sensor τ → delay and breath-by-breath accuracy.
13. Summary table and caveats.

## Smoke-run highlights (simulation, not measurements)
- **Ideal sensors:** textbook Haldane recovers the true VO₂ to 0.0 % mean error. The patent's eq. 4 with dry-air F_IO₂ also recovers it. Combining eq. 4 with the patent's printed F_IO₂ = 0.2005 gives −17.5 % at rest and −22 % on the ramp.
- **Default hardware** (1 m × 1.2 mm tube, 150 mL/min, τ 0.10/0.15 s): the delay is recovered as 581 ms against a simulated 598 ms. VO₂ error is −1.1 % at rest and −5.7 % on average over the ramp, growing at high breathing rates.
- **VO₂ bias at 45 breaths/min**, every breath analysed: −0.9 % with τ = 0.05 s, −4.4 % with τ = 0.1 s, −60 % with τ = 0.8 s. With the slow sensor:
  - skip N = 3 + sampling chamber + constant pump: −9.9 %
  - skip N = 3 + flow-proportional pump: −0.2 %
  - software speed-up: +0.4 %
- **No dryer**, device assuming ambient humidity: about +18 % VO₂ error.
- **Leaks:** a 10 % expiratory leak gives −10 % at rest. A 2 % inspiratory leak gives −8 % if VI is integrated directly, and none with Haldane.
- **Thresholds:** VT1 by V-slope 1.57 L/min (truth 1.60). VT2 2.64 L/min (truth 2.80); the device VO₂ under-reads at high rates, but in ramp time it is within 0.1 min.

## Open-source VO₂ projects (verified 23 Nov 2026)
None of these uses a pumped sidestream like the patent's FIG. 1. All put the gas sensors inline in or beside the exhaled flow path.

| Project | Sensors / approach |
|---|---|
| [meteoscientific/VO2max](https://github.com/meteoscientific/VO2max) (fork of Ivor Hewitt's build, with Ulrich Rissel's CO₂ upgrade) | ESP32 TTGO T-Display, DFRobot Gravity I²C O₂ (electrochemical), Sensirion SCD30 CO₂, BMP280, venturi + differential-pressure flow; 3M mask; < $200 target. Work in progress (README dated Aug 2023). |
| [Rabbitcreek, "Accurate VO2 Max for Zwift and Strava"](https://www.instructables.com/Accurate-VO2-Max-for-Zwift-and-Strava/) (Instructables, the original) | ESP32, SDP816-125Pa differential-pressure venturi flow, DFRobot O₂, SCD30 CO₂ add-on; one-way-valve snorkel so only exhaled gas is measured. The author reports ±8 % volume accuracy against a home-made 1 L syringe. |
| [salbito/VO2max](https://github.com/salbito/VO2max) | Copy of Ivor Hewitt's code and build files (the original `ihewitt/VO2max` repo now returns 404). |
| [zanppa/VO2Max](https://github.com/zanppa/VO2Max) | Rewrite of Hewitt's build: FreeRTOS, 20 ms flow sampling, O₂, ambient P/T for STPD, VE and VO₂ only (no CO₂), WiFi streaming. Prototype. |
| [WeiyuanWang0821/Real-Time-Cycling-Monitor](https://github.com/WeiyuanWang0821/Real-Time-Cycling-Monitor) | Raspberry Pi 5 + Qt: 4MZ-HH O₂, Sensirion SCD4x CO₂ (the README says it is too slow for real-time, about 60 s), Sensirion SFM3300 flow; custom mask with a one-way valve. |
| [stephenverderame/Calorimeter](https://github.com/stephenverderame/Calorimeter) | Arduino + ESP8266, K30 CO₂, "Wind Sensor Rev C" hot-wire anemometer; streams over UDP. Early-stage. |

**Analysis libraries** (no hardware): [ropensci/spiro](https://github.com/ropensci/spiro) and [fmmattioni/whippr](https://github.com/fmmattioni/whippr), R packages for breath-by-breath CPET data.
