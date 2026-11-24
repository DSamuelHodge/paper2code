# ---- core simulation library (cell 1) ----
import numpy as np, math
from scipy.signal import lfilter
FS = 100.0                     # 🧪 simulation sample rate, Hz
DT = 1.0 / FS
# 📚 textbook / standard-atmosphere constants
P_STD = 1013.25                # hPa
T0 = 273.15
FiO2_DRY = 0.2095              # 📚 dry air O2 fraction
FiCO2_DRY = 0.00042            # 📚 ~420 ppm (NOAA global mean, early 2020s)
def p_sat(T_c):                # 📚 Magnus formula (Alduchov & Eskridge 1996), hPa
    return 6.1094 * np.exp(17.625 * T_c / (T_c + 243.04))
def to_stpd_factor(PB, T_c, pH2O):   # 📚 V_STPD,dry = V * (PB - pH2O)/1013.25 * 273.15/(273.15+T)
    return (PB - pH2O) / P_STD * T0 / (T0 + T_c)
# 📜 constants printed in the patent (US 2019/0120821 A1, [0099]-[0121])
PAT_FEN2_SUM = 0.99063
PAT_FIN2 = 0.7808
PAT_FIO2 = 0.2005
PAT_FICO2 = 0.00039

class Env:                     # 🧪 room conditions for the simulation
    PB = 1013.25; T_amb = 22.0; RH_amb = 0.40
    T_exp_at_flow = 34.0       # 🧪 expired gas cools from 37 °C in the mask; saturated
    @property
    def pH2O_amb(self): return self.RH_amb * p_sat(self.T_amb)

# ---- 🧪 ground-truth physiology (assumed, typical-looking numbers, NOT from the patent) ----
class Truth:
    def __init__(self, rest_s=300, ramp_s=720, ramp_w_per_min=25, vo2_rest=0.30, vo2_unloaded=0.55,
                 slope_l_per_w=0.010, mrt=40.0, vt1=1.6, vt2=2.8, rer_rest=0.82, s1=0.95, s2=1.30,
                 vevco2=28.0, vevco2_rise=7.0):
        self.__dict__.update(locals()); del self.__dict__["self"]
        self.T = rest_s + ramp_s
        tg = np.arange(0, self.T + 60, 1.0)
        W = np.where(tg < rest_s, 0.0, (tg - rest_s) / 60 * ramp_w_per_min)
        target = np.where(tg < rest_s, vo2_rest, vo2_unloaded + slope_l_per_w * W)
        vo2 = np.empty_like(tg); vo2[0] = vo2_rest
        for i in range(1, len(tg)):
            vo2[i] = vo2[i-1] + (target[i] - vo2[i-1]) / mrt
        self.tg, self.W, self.vo2_g = tg, W, vo2
    def vco2_of_vo2(self, vo2):
        a = (self.rer_rest - self.s1) * self.vo2_rest
        v1 = a + self.s1 * self.vt1
        return np.where(vo2 < self.vt1, a + self.s1 * vo2, v1 + self.s2 * (vo2 - self.vt1))
    def ve_of(self, vo2, vco2):  # L/min BTPS
        ratio = self.vevco2 + self.vevco2_rise * np.clip(vo2 - self.vt2, 0, None)
        return vco2 * ratio
    def state(self, t):
        vo2 = np.interp(t, self.tg, self.vo2_g)
        vco2 = self.vco2_of_vo2(vo2); ve = self.ve_of(vo2, vco2)
        f = np.clip(12 + 36 * (ve - 8) / (130 - 8), 12, 50)
        ti_frac = 0.40 + 0.08 * (f - 12) / 36
        return dict(vo2=vo2, vco2=vco2, ve=ve, f=f, ti_frac=ti_frac, W=np.interp(t, self.tg, self.W))

def gen_breaths(duration, state_fn, env=Env(), seed=0, cv_vt=0.08, cv_T=0.06, VD=0.15,
                FiO2=FiO2_DRY, FiCO2=FiCO2_DRY):
    """Mouth-level signals. flow: L/s STPD-dry (+ = expiration). FO2/FCO2: DRY fractions at the mouth."""
    rng = np.random.default_rng(seed)
    k_btps = to_stpd_factor(env.PB, 37.0, p_sat(37.0))
    FIN = 1 - FiO2 - FiCO2
    flow, fo2, fco2, phase, bidx, rows = [], [], [], [], [], []
    t, k = 0.0, 0
    while t < duration:
        s = state_fn(t)
        Ttot = 60 / s["f"] * rng.lognormal(0, cv_T)
        Ti = s["ti_frac"] * Ttot
        ni, ne = max(int(round(Ti * FS)), 5), max(int(round((Ttot - Ti) * FS)), 5)
        Ttot = (ni + ne) / FS
        VE = s["ve"] / s["f"] * k_btps * rng.lognormal(0, cv_vt)   # expired L STPD-dry this breath
        VO2b, VCO2b = s["vo2"] * Ttot / 60, s["vco2"] * Ttot / 60
        te = (np.arange(ne) + 0.5) / ne
        qe = np.sin(np.pi * te); qe *= VE / (qe.sum() * DT)
        v = np.cumsum(qe) * DT
        sh = 1 / (1 + np.exp(-(v - VD) / 0.03)) * (0.92 + 0.08 * v / VE)   # capnogram phases I-III
        S = (qe * sh).sum() * DT
        A = np.array([[S, 0, -FiCO2], [0, S, FiO2], [S, -S, FIN]])
        b = np.array([VCO2b - FiCO2 * VE, VO2b + FiO2 * VE, VE * FIN])
        dC, dO, VI = np.linalg.solve(A, b)
        ti_ = (np.arange(ni) + 0.5) / ni
        qi = np.sin(np.pi * ti_); qi *= VI / (qi.sum() * DT)
        flow += [-qi, qe]
        fo2 += [np.full(ni, FiO2), FiO2 - dO * sh]
        fco2 += [np.full(ni, FiCO2), FiCO2 + dC * sh]
        phase += [-np.ones(ni), np.ones(ne)]
        bidx += [np.full(ni + ne, k)]
        rows.append((k, t, t + Ttot, t + Ti, Ttot, VO2b, VCO2b, VE, VI, s["vo2"], s["vco2"]))
        t += Ttot; k += 1
    cat = np.concatenate
    sig = dict(t=np.arange(len(cat(flow))) * DT, flow=cat(flow), fo2=cat(fo2), fco2=cat(fco2),
               phase=cat(phase), bidx=cat(bidx).astype(int))
    cols = ["k", "t0", "t1", "t_exp", "Ttot", "VO2b", "VCO2b", "VEb", "VIb", "vo2_rate", "vco2_rate"]
    br = {c: np.array([r[i] for r in rows]) for i, c in enumerate(cols)}
    return sig, br

# ---- sidestream + sensors (cell 2) ----
def first_order(x, tau):
    if tau <= 0: return x.copy()
    a = math.exp(-DT / tau)
    y = lfilter([1 - a], [1, -a], x - x[0]) + x[0]
    return y
def delay(x, d_s):
    if d_s <= 0: return x.copy()
    t = np.arange(len(x)) * DT
    return np.interp(t - d_s, t, x, left=x[0])

class Hardware:
    """🧪 Sidestream hardware knobs (the defaults are simulation assumptions, not patent numbers)."""
    def __init__(self, tube_len_m=1.0, tube_id_mm=1.2, pump_ml_min=150.0, v_between_ml=0.3,
                 tau_co2=0.10, tau_o2=0.15, mix_frac=0.10, dryer="nafion", T_sensor=35.0,
                 noise_o2=2e-4, noise_co2=1e-4, noise_flow=0.01, o2_gain_drift=0.0, co2_offset_drift=0.0,
                 leak_exp=0.0, leak_insp=0.0, leak_mode="const"):
        self.__dict__.update(locals()); del self.__dict__["self"]
    @property
    def v_tube_ml(self): return math.pi * (self.tube_id_mm / 2e3) ** 2 * self.tube_len_m * 1e6
    @property
    def transport_delay(self): return self.v_tube_ml / self.pump_ml_min * 60.0
    @property
    def between_delay(self): return self.v_between_ml / self.pump_ml_min * 60.0

def measure(sig, hw, env=Env(), seed=1, skip_mask=None):
    """Turn mouth-level truth into what the device's sensors report."""
    rng = np.random.default_rng(seed)
    n = len(sig["t"]); t = sig["t"]
    # --- flow sensor at the mask (actual volume flow at local conditions) + leaks
    q = sig["flow"].copy()
    exp_ = q > 0
    if hw.leak_mode == "const":
        q = np.where(exp_, q * (1 - hw.leak_exp), q * (1 - hw.leak_insp))
    else:  # leak grows with flow (pressure) 🧪
        qmax = np.abs(q).max()
        q = np.where(exp_, q * (1 - hw.leak_exp * 2 * q / qmax), q * (1 - hw.leak_insp * 2 * np.abs(q) / qmax))
    k_exp = to_stpd_factor(env.PB, env.T_exp_at_flow, p_sat(env.T_exp_at_flow))
    k_insp = to_stpd_factor(env.PB, env.T_amb, env.pH2O_amb)
    q_meas = np.where(exp_, q / k_exp, q / k_insp)              # L/s at local conditions
    q_meas = q_meas * (1 + hw.noise_flow * rng.standard_normal(n))
    # --- gas drawn into the sidestream line: dry fractions + water
    if hw.dryer == "nafion":     # 📜 Nafion brings the sample to ambient RH ([0065],[0109])
        xw = np.full(n, env.pH2O_amb / env.PB)
    else:                        # no dryer: expired gas saturated at sensor T, inspired at ambient
        xw_exp = min(p_sat(37.0), p_sat(hw.T_sensor)) / env.PB
        xw = np.where(sig["phase"] > 0, xw_exp, env.pH2O_amb / env.PB)
    line = {}
    td = hw.transport_delay; tau_mix = hw.mix_frac * td
    for name in ("fo2", "fco2"):
        g = sig[name].copy()
        if skip_mask is not None:            # 🧪 purge with ambient during skipped breaths
            amb = FiO2_DRY if name == "fo2" else FiCO2_DRY
            g = np.where(skip_mask, amb, g)
        line[name] = first_order(delay(g, td), tau_mix)
    xw_l = first_order(delay(xw, td), tau_mix)
    # in-line order: CO2 sensor first, then O2 sensor (📜 either order allowed, [0059])
    co2_wet = line["fco2"] * (1 - xw_l)
    o2_wet = delay(line["fo2"] * (1 - xw_l), hw.between_delay)
    co2_s = first_order(co2_wet, hw.tau_co2)
    o2_s = first_order(o2_wet, hw.tau_o2)
    frac = t / t[-1]
    o2_s = o2_s * (1 + hw.o2_gain_drift * frac) + hw.noise_o2 * rng.standard_normal(n)
    co2_s = co2_s + hw.co2_offset_drift * frac + hw.noise_co2 * rng.standard_normal(n)
    xw_rh = first_order(xw_l, 1.0)     # 🧪 in-line humidity sensor, tau = 1 s, reported as water mole fraction
    return dict(t=t, flow=q_meas, o2=o2_s, co2=co2_s, xw_rh=xw_rh, xw_true=xw_l)

# ---- device-side processing (cell 3) ----
def detect_breaths(flow, thr=0.03, min_phase_s=0.25):
    """Inspiration starts when flow < -thr, expiration starts when flow > +thr (hysteresis)."""
    state, starts = 0, []          # list of (index, +1 exp / -1 insp)
    last = -10**9
    for i, q in enumerate(flow):
        if state <= 0 and q > thr and i - last > min_phase_s * FS:
            state, last = 1, i; starts.append((i, 1))
        elif state >= 0 and q < -thr and i - last > min_phase_s * FS:
            state, last = -1, i; starts.append((i, -1))
    # breath = insp start -> exp start -> next insp start
    out = []
    for j in range(len(starts) - 2):
        (a, sa), (b, sb), (c, sc) = starts[j], starts[j+1], starts[j+2]
        if sa == -1 and sb == 1 and sc == -1:
            out.append((a, b, c))
    return np.array(out, dtype=int)

def estimate_gas_delay(m, breaths, tau_co2_datasheet):
    """🧪 our method. Find the time from each exp->insp flow reversal to the 50 % fall of the CO2 trace
    (the inspiratory edge is sharp because fresh air replaces alveolar gas at once). For a first-order
    sensor the 50 % point lags by tau*ln2 but the signal's centre of mass lags by tau, so we align by
    t50 + tau*(1 - ln2)."""
    co2, ds = m["co2"], []
    lo = np.percentile(co2, 2)
    for a, b, c in breaths[5:-5]:
        hi = co2[b:c + int(3 * FS)].max()
        if hi - lo < 0.005: continue
        mid = lo + 0.5 * (hi - lo)
        w = co2[c:c + int(3 * FS)]
        # first downward crossing of the midpoint after the flow reversal (valid while delay < breath period)
        idx = np.where((w[:-1] >= mid) & (w[1:] < mid))[0]
        if len(idx): ds.append((idx[0] + 1) * DT)
    t50 = float(np.median(ds)) if ds else 0.0
    return dict(t50=t50, align=t50 + tau_co2_datasheet * (1 - math.log(2)))

def fit_o2_vs_co2(m, max_lag_s=0.6, taus=np.arange(0, 1.01, 0.05)):
    """📜 patent eq.(5): fO2(t) = -c1*fCO2(t) + c2. 🧪 we search the lag (and extra lag-filter) that best fits it."""
    o2, co2 = m["o2"], m["co2"]
    best = None
    for tau in taus:
        c = first_order(co2, tau)
        for L in range(0, int(max_lag_s * FS) + 1):
            x, y = c[: len(c) - L], o2[L:]
            X = np.vstack([-x, np.ones_like(x)]).T
            coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
            r2 = 1 - res[0] / ((y - y.mean()) ** 2).sum()
            if best is None or r2 > best[0]: best = (r2, L * DT, tau, coef)
    return dict(r2=best[0], lag_s=best[1], extra_tau_s=best[2], c1=best[3][0], c2=best[3][1])

def speed_up(x, tau, smooth_s=0.06):
    """🧪 invert a first-order sensor: x_fast = x + tau*dx/dt (derivative from a Savitzky-Golay filter)."""
    from scipy.signal import savgol_filter
    w = max(5, int(smooth_s * FS) | 1)
    dx = savgol_filter(x, w, 2, deriv=1, delta=DT)
    return savgol_filter(x, w, 2) + tau * dx

def shift(x, d_s):  # advance a signal by d_s seconds (undo a delay)
    t = np.arange(len(x)) * DT
    return np.interp(t + d_s, t, x, right=x[-1])

def bbb(m, breaths, env=Env(), gas_delay=0.0, o2_extra_lag=0.0, dryer_assumed="nafion",
        FiO2=None, FiCO2=None, autozero=False, method="haldane", haldane="textbook", use_rh=False,
        exp_as="btps", speedup_tau=(0.0, 0.0)):
    """Breath-by-breath VO2/VCO2. Returns dict of arrays (L/min STPD)."""
    k_insp = to_stpd_factor(env.PB, env.T_amb, env.pH2O_amb)
    if exp_as == "btps": k_exp = to_stpd_factor(env.PB, env.T_exp_at_flow, p_sat(env.T_exp_at_flow))
    else: k_exp = k_insp            # mistake on purpose: treat exhaled gas as if it were room air (ATPS)
    q = np.where(m["flow"] > 0, m["flow"] * k_exp, m["flow"] * k_insp)   # back to STPD-dry
    co2_raw, o2_raw = m["co2"], m["o2"]
    if speedup_tau[0] > 0: co2_raw = speed_up(co2_raw, speedup_tau[0])
    if speedup_tau[1] > 0: o2_raw = speed_up(o2_raw, speedup_tau[1])
    co2 = shift(co2_raw, gas_delay); o2 = shift(o2_raw, gas_delay + o2_extra_lag)
    if use_rh: xw = shift(m["xw_rh"], gas_delay)
    else: xw = np.full(len(q), env.pH2O_amb / env.PB)     # assume sample at ambient humidity (dryer)
    o2d, co2d = o2 / (1 - xw), co2 / (1 - xw)              # wet -> dry fractions
    rows = []
    if autozero:   # 📜 ambient air between exhalations is "useful for calibration" ([0072])
        # 🧪 our method: per breath, the top 20 % of the aligned O2 trace (and bottom 20 % of CO2) during
        # inspiration is taken as ambient; a running median over 9 breaths suppresses noise
        amb_o2, amb_co2 = [], []
        for a, b, c in breaths:
            wo, wc = np.sort(o2d[a:b]), np.sort(co2d[a:b]); k5 = max(1, len(wo) // 5)
            amb_o2.append(wo[-k5:].mean()); amb_co2.append(wc[:k5].mean())
        from scipy.ndimage import median_filter
        amb_o2 = median_filter(np.array(amb_o2), 9, mode="nearest"); amb_co2 = median_filter(np.array(amb_co2), 9, mode="nearest")
    for j, (a, b, c) in enumerate(breaths):
        if autozero: g_o2, off_co2 = FiO2_DRY / amb_o2[j], amb_co2[j] - FiCO2_DRY
        else: g_o2, off_co2 = 1.0, 0.0
        fo = o2d[b:c] * g_o2; fc = co2d[b:c] - off_co2
        qe = q[b:c].clip(min=0); VE = qe.sum() * DT
        FEO2 = (qe * fo).sum() * DT / VE; FECO2 = (qe * fc).sum() * DT / VE
        fio2 = FiO2_DRY if FiO2 is None else FiO2; fico2 = FiCO2_DRY if FiCO2 is None else FiCO2
        if method == "haldane":
            if haldane == "textbook":
                VI = VE * (1 - FEO2 - FECO2) / (1 - fio2 - fico2)
            else:   # 📜 patent eq.(4) constants
                VI = VE * (PAT_FEN2_SUM - (FEO2 + FECO2)) / PAT_FIN2
        else:       # integrate the measured inspired flow directly
            VI = -q[a:b].clip(max=0).sum() * DT
        Ttot = (c - a) * DT
        VO2 = (VI * fio2 - VE * FEO2) / Ttot * 60
        VCO2 = (VE * FECO2 - VI * fico2) / Ttot * 60
        rows.append(((a + c) / 2 * DT, b, VO2, VCO2, VE / Ttot * 60, Ttot, FEO2, FECO2))
    cols = ["t", "i_exp", "VO2", "VCO2", "VE", "Ttot", "FEO2", "FECO2"]
    return {cc: np.array([r[i] for r in rows]) for i, cc in enumerate(cols)}

def match_truth(res, sig, br):
    """Pair each detected breath with the true breath that owns its expiration start."""
    kk = sig["bidx"][np.clip(res["i_exp"].astype(int) + 2, 0, len(sig["bidx"]) - 1)]
    tv = br["VO2b"][kk] / br["Ttot"][kk] * 60; tc = br["VCO2b"][kk] / br["Ttot"][kk] * 60
    return tv, tc

def err_stats(res, sig, br, skip_first=3):
    tv, tc = match_truth(res, sig, br)
    e = (res["VO2"] - tv) / tv * 100; ec = (res["VCO2"] - tc) / tc * 100
    e, ec = e[skip_first:-1], ec[skip_first:-1]
    return dict(vo2_bias=e.mean(), vo2_sd=e.std(), vco2_bias=ec.mean(), vco2_sd=ec.std(), n=len(e))

# ---- helpers for later sections (cell 4) ----
def sample_and_hold(sig, breaths, N, td, tau, proportional=False):
    """💭/🧪 one reading of the patent's 'analyse every Nth breath' mode combined with its optional
    sampling chamber: the analysed breath's sample is collected into a small chamber (purged before each
    fill, an assumption), the valve then isolates it and the slow sensor settles during the N-1 skipped
    breaths. Pump flow is constant, or proportional to expiratory flow (proportional=True)."""
    n = len(sig["t"]); q = np.clip(sig["flow"], 0, None)
    gin = {k: delay(sig[k], td) for k in ("fo2", "fco2")}
    qs = delay(q, td) if proportional else np.ones(n)
    cham = {k: np.full(n, FiO2_DRY if k == "fo2" else FiCO2_DRY) for k in gin}
    reads = []
    an = list(range(0, len(breaths) - N, N))
    for j, bi in enumerate(an):
        a, b, c = breaths[bi]
        lo, hi = b + int(td * FS), c + int(td * FS)
        nxt = breaths[an[j + 1]][1] + int(td * FS) if j + 1 < len(an) else n
        hi, nxt = min(hi, n), min(nxt, n)
        for k in gin:
            w = qs[lo:hi]; cs = np.cumsum(w * gin[k][lo:hi]) / np.maximum(np.cumsum(w), 1e-12)
            cham[k][lo:hi] = cs; cham[k][hi:nxt] = cs[-1] if len(cs) else cham[k][lo]
        reads.append((bi, nxt - 1))
    sens = {k: first_order(cham[k], tau) for k in cham}
    out = []
    for bi, i_read in reads:
        a, b, c = breaths[bi]
        out.append((bi, sens["fo2"][i_read], sens["fco2"][i_read]))
    return out

def two_segment(x, y, lo_frac=0.15, hi_frac=0.85):
    """Best single breakpoint for a continuous two-line fit (least squares)."""
    o = np.argsort(x); x, y = x[o], y[o]
    best = None
    for i in range(int(lo_frac * len(x)), int(hi_frac * len(x))):
        xb = x[i]
        X = np.vstack([np.ones_like(x), x, np.clip(x - xb, 0, None)]).T
        coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = ((X @ coef - y) ** 2).sum()
        if best is None or sse < best[0]: best = (sse, xb, coef)
    return best[1], best[2]

def bin_avg(t, ys, width=10.0):
    edges = np.arange(t.min(), t.max() + width, width); idx = np.digitize(t, edges)
    keep = [i for i in np.unique(idx) if (idx == i).sum() >= 2]
    return np.array([t[idx == i].mean() for i in keep]), [np.array([y[idx == i].mean() for i in keep]) for y in ys]
