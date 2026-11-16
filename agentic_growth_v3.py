"""
Super-Exponential Growth in the Agentic Era -- model v3.

Changes from v2 (second review):
  * Construction lead time tau: machine-built fabs, robots and power plants ordered at t arrive at t + tau.
    Delay-differential equations, solved by the method of steps. tau = 0 reproduces v2.
  * I_M is physical, not cash: the share s_M of machine labor (agent-hours and robot-hours) redirected to
    construction, converted to capital at 2026 unit costs. Wright's law lowers those unit costs as cumulative
    machine-built capital C grows: dC/dt = completed machine-built capital.
  * Human-built investment funds chips and fabs only (shares renormalised); human-built robots follow the
    IFR + BofA humanoid path and human-built power grows at g_P. Stated explicitly.
  * Power rationing stated: when K w_K + R w_R reaches P, compute and robots each grow no faster than P
    (compute also gains the efficiency trend).
  * Three mechanisms are separable: intelligence amplification (g_E, psi), capital self-replication (a(t)),
    production learning (LR). Scenarios isolate each.
  * s_0 recalibrated to 0.8% of GDP (2026 AI capex ~ $1T incl. non-hyperscalers).
  * BofA humanoid path: 90k, 290k, 500k, 800k, 1.2M (2026-2030), then +50%/yr.
  * v3.1: human-built AI power is an additions path, not a growth rate: A_H(t) = A0 exp(gA t), A0 = 30 GW/yr in 2026,
    gA = ln 1.4 (SemiAnalysis Oct-2026: 25 GW net IT capacity 2026, 51 GW 2027; Patel Aug-2026: 30/50/70/90-100 GW
    for 2026-29, ~200 GW cumulative by end-2028). s_max raised (Patel: hyperscaler capex $1T 2026, >$2T 2028).

  * Stopping rule: a run ends when ln(K/K0) > 40 or dlnK/dt > 50/yr. It is reported as a finite-time blow-up
    only when tau = 0 and the growth rate at the stop exceeds 20/yr (a double-exponential path crosses e^40 at
    about psi*gE*40 ~ 11/yr). A delayed system is finite at every finite t, so it is never reported as a blow-up.
  * The delayed term switches on continuously at t = tau; C carries an absolute tolerance of $1 (it is in dollars).

Stocks (logs): K compute [H100e], F fab capacity [H100e/yr], R robots, P AI power [GW]; C cumulative
machine-built capital [$ at 2026 prices] in linear space.  t = 0 is 1 Jan 2026.
    python3 agentic_growth_v3.py            # all scenarios -> v3_summary.csv
    python3 v3_sens.py                      # grids in section 6.3
"""
import math, csv, sys
import numpy as np
from scipy.integrate import solve_ivp

BASE = dict(
    Y0=125e12, YA0=0.05e12, K0=20.3e6, F0=18e6, R0=5.08e6, P0=22.0,
    vR=25_000.0, pC0=37_000.0, gp=math.log(1.37), pF=18_000.0, pR=100_000.0, pP=3e9,
    dK=0.20, dF=0.10, dR=0.08,
    s0=0.008, sM=0.30, sK=0.50, sF=0.20, sR=0.15, sP=0.15,
    w0=1.1, gw=math.log(1.4), wR=1.0, A0=30.0, gA=math.log(1.4), Cref=100e9, c0=0.1, phi=0.6,
    ind0=0.655e6, ind_g=math.log(806/655)/3,
    hum=[0.09e6, 0.29e6, 0.50e6, 0.80e6, 1.20e6], hum_g=math.log(1.5),   # BofA 2026-2030, then +50%/yr
)
SCEN = {
    "slow":     dict(rho=0.5,  gE=math.log(2.0), gF=math.log(1.5), smax=0.012, Ta=12.0, wa=1.5, LR=0.05, psi=0.2, mP=1.0, absorb=None, disp=0.0, tau=1.0),
    "baseline": dict(rho=0.65, gE=math.log(2.5), gF=math.log(1.7), smax=0.020, Ta=7.0,  wa=1.0, LR=0.15, psi=0.3, mP=1.0, absorb=None, disp=0.0, tau=1.0),
    "fast":     dict(rho=0.8,  gE=math.log(3.0), gF=math.log(2.0), smax=0.035, Ta=5.0,  wa=1.0, LR=0.25, psi=0.4, mP=1.0, absorb=None, disp=0.0, tau=1.0),
}
B = SCEN["baseline"]
SCEN["baseline_tau0"]    = {**B, "tau": 0.0}
SCEN["baseline_tau2"]    = {**B, "tau": 2.0}
SCEN["trend_only"]       = {**B, "Ta": 80.0}                       # no self-replication
SCEN["replication_only"] = {**B, "LR": 0.0, "psi": 0.0}            # a(t) on, no learning, robots not amplified
SCEN["repl_plus_intel"]  = {**B, "LR": 0.0}                        # + intelligence amplification (psi)
SCEN["repl_plus_learn"]  = {**B, "psi": 0.0}                       # + production learning (LR)
SCEN["human_energy"]     = {**B, "mP": 0.0}
SCEN["absorption_bound"] = {**B, "absorb": math.log(2.0)}
SCEN["displacement"]     = {**B, "disp": 0.5}
SCEN["power_backlash"]   = {**B, "gA": math.log(1.25)}             # datacentre moratoria: human-built power additions +25%/yr

T_END, T_MAX, BLOW, RATE_STOP, MAX_STEP = 10.0, 22.0, 40.0, 50.0, 0.01   # stop at ln(K/K0) > 40 or at dlnK/dt > 50/yr

def human_gdp(t, p):
    base = p["Y0"] - p["YA0"] - p["vR"] * p["R0"]
    return base * math.exp(0.031 * min(t, 1) + 0.034 * min(max(t - 1, 0), 1) + 0.031 * max(t - 2, 0))
def logistic(t, m, w): return 1 / (1 + math.exp(-(t - m) / w))
def s_H(t, p):
    L0, L = logistic(0, 2.5, 1.2), logistic(t, 2.5, 1.2)
    return p["s0"] + (p["smax"] - p["s0"]) * (L - L0) / (1 - L0)
def humanoids(t, p):
    h = p["hum"]
    if t >= 4: return h[4] * math.exp(p["hum_g"] * (t - 4))
    i = int(math.floor(t)); f = t - i
    return h[i] * (h[i + 1] / h[i]) ** f
def robots_human(t, p): return p["ind0"] * math.exp(p["ind_g"] * t) + humanoids(t, p)
def wright(C, p): return (1 + C / p["Cref"]) ** (-math.log2(1 - p["LR"])) if p["LR"] > 0 else 1.0

def outputs(t, y, p):
    K, F, R, P = map(math.exp, y[:4]); C = y[4]          # C integrated in linear space
    E = math.exp(p["gE"] * t)
    YH = human_gdp(t, p)
    YA = p["YA0"] * ((K / p["K0"]) * E) ** p["rho"]
    YR = p["vR"] * R * E ** p["psi"]
    if p["absorb"] is not None:
        cap = (p["YA0"] + p["vR"] * p["R0"]) * math.exp(p["absorb"] * t)
        if YA + YR > cap:
            f = cap / (YA + YR); YA, YR = YA * f, YR * f
    YH = max(YH - p["disp"] * (YA + YR - p["YA0"] - p["vR"] * p["R0"]), 0.2 * YH)
    return K, F, R, P, C, E, YH, YA, YR, logistic(t, p["Ta"], p["wa"])

def machine_orders(t, y, p):
    """Machine construction at t, $ of capital at 2026 unit costs: automated share of machine labor, learning-boosted."""
    K, F, R, P, C, E, YH, YA, YR, a = outputs(t, y, p)
    return a * wright(C, p) * p["sM"] * (YA + YR)

def rhs(t, y, p, hist):
    K, F, R, P, C, E, YH, YA, YR, a = outputs(t, y, p)
    pC = p["pC0"] * math.exp(-p["gp"] * t)
    I_h = s_H(t, p) * YH + (1 - a) * p["sM"] * (YA + YR)              # human-built: chips and fabs only
    I_m_now = machine_orders(t, y, p)
    tau = p["tau"]
    I_m_done = I_m_now if tau == 0 else (machine_orders(t - tau, hist(t - tau), p) if t - tau >= 0 else 0.0)
    hK, hF = p["sK"] / (p["sK"] + p["sF"]), p["sF"] / (p["sK"] + p["sF"])
    dF = min(I_h * hF / p["pF"], F * (p["gF"] + p["dF"])) + I_m_done * p["sF"] / p["pF"] - p["dF"] * F
    dK = min((I_h * hK + I_m_done * p["sK"]) / pC, F) - p["dK"] * K
    dR = robots_human(t, p) + I_m_done * p["sR"] / p["pR"] - p["dR"] * R
    dP = p["A0"] * math.exp(p["gA"] * t) + p["mP"] * I_m_done * p["sP"] / p["pP"]   # human-built additions path + machine-built
    kW = p["w0"] * math.exp(-p["gw"] * t)
    if K * kW * 1e-6 + R * p["wR"] * 1e-6 >= 0.999 * P:                  # power rationing
        g = dP / P
        dK = min(dK, K * (g + p["gw"])); dR = min(dR, R * g)
    dC = I_m_done
    return [dK / K, dF / F, dR / R, dP / P, dC]

def run(name, over=None, t_max=T_MAX):
    p = {**BASE, **SCEN[name], **(over or {})}
    y0 = [math.log(p["K0"]), math.log(p["F0"]), math.log(p["R0"]), math.log(p["P0"]), 0.0]   # C0 = 0
    tau = p["tau"]; chunks = []
    def hist(t):
        if not chunks: return y0
        for (t0, t1, sol) in chunks:
            if t0 - 1e-12 <= t <= t1 + 1e-12: return sol.sol(t)
        return chunks[-1][2].sol(t)
    def blow(t, y): return BLOW - (y[0] - math.log(p["K0"]))
    def fast(t, y): return RATE_STOP - rhs(t, y, p, hist)[0]
    blow.terminal = fast.terminal = True
    t, y, t_star, rate_at_stop = 0.0, y0, None, None
    step = tau if tau > 0 else t_max
    while t < t_max - 1e-9:
        t1 = min(t + step, t_max)
        sol = solve_ivp(lambda tt, yy: rhs(tt, yy, p, hist), (t, t1), y, events=[blow, fast], max_step=MAX_STEP,
                        rtol=1e-7, atol=[1e-9] * 4 + [1.0], dense_output=True, method="RK45")   # C is in dollars
        chunks.append((t, sol.t[-1], sol))
        hit = [(ev[0], i) for i, ev in enumerate(sol.t_events) if len(ev)]
        if hit:
            te, i = min(hit); rate_at_stop = rhs(te, sol.y_events[i][0], p, hist)[0]
            # finite-time blow-up: no delay, and the growth rate is already exploding when K has grown e^40-fold
            # (a double-exponential path crosses e^40 at a rate of only ~psi*gE*40 ~ 11/yr). A delayed system never blows up.
            if tau == 0 and (i == 1 or rate_at_stop > 20): t_star = te
            t = te; break
        t, y = sol.t[-1], sol.y[:, -1]
    t_last = t
    def st(tt):
        K, F, R, P, C, E, YH, YA, YR, a = outputs(tt, hist(tt), p)
        kW = p["w0"] * math.exp(-p["gw"] * tt)
        return dict(year=2025 + tt, Y=YH + YA + YR, YH=YH, YA=YA, YR=YR, K=K, F=F, R=R, P=P, C=C, a=a, W=wright(C, p),
                    GW=K * kW * 1e-6 + R * p["wR"] * 1e-6, mshare=(YA + YR) / (YH + YA + YR))
    def growth(tt): return st(tt)["Y"] / st(tt - 1)["Y"] - 1
    g30 = mach = None
    for tt in np.arange(1.0, t_last, 0.01):
        s = st(tt)
        if g30 is None and growth(tt) > 0.30: g30 = 2026 + tt
        if mach is None and s["YA"] + s["YR"] > s["YH"]: mach = 2026 + tt
    return p, st, growth, t_star, t_last, g30, mach

if __name__ == "__main__":
    names = sys.argv[1:] or list(SCEN)
    out = []
    for n in names:
        p, st, growth, ts, tl, g30, mach = run(n)
        row = dict(scenario=n, tau=p["tau"])
        print(f"\n== {n} (tau={p['tau']}) ==")
        for yr in [2026, 2028, 2030, 2032, 2033, 2034, 2035, 2037, 2040, 2045, 2050]:
            tt = yr - 2025
            if tt > tl: break
            s = st(tt); g = growth(tt)
            row[f"g{yr}"] = g
            print(f"end{yr} Y={s['Y']/1e12:9.1f}T YH={s['YH']/1e12:6.1f} YA={s['YA']/1e12:9.2f} YR={s['YR']/1e12:9.2f} g={g*100:8.1f}% "
                  f"K={s['K']/1e6:9.1f}M R={s['R']/1e6:8.2f}M P={s['P']:8.0f}GW use={s['GW']:6.0f} a={s['a']:.2f} W={s['W']:.2f}")
        s30 = st(5.0)
        row.update(Y2030=s30["Y"]/1e12, YA2030=s30["YA"]/1e12, YR2030=s30["YR"]/1e12, K2030=s30["K"]/1e6, R2030=s30["R"]/1e6, GW2030=s30["GW"],
                   g30=g30, mach=mach, blowup=(2026 + ts) if ts else None, stop=2026 + tl, K_sep26=st(0.75)["K"]/1e6, YA_end26=st(1.0)["YA"]/1e9)
        print(f"  >30%: {g30}  machine>human: {mach}  blowup: {row['blowup']}  stopped: {row['stop']:.1f}")
        out.append(row)
    keys = sorted({k for r in out for k in r}, key=lambda k: (k not in ("scenario", "tau"), k))
    with open("v3_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
