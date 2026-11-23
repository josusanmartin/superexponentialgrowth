"""Super-Exponential Growth in the Agentic Era: the full four-stock model.

Stocks: compute K [H100e], fab capacity F [H100e/yr], robots R, AI power P [GW], all in logs, and cumulative
commissioned machine construction C [$ at 2026 prices] in levels. t = 0 is 1 January 2026.

Machine-built capital ordered at t enters service at t + tau and earns experience only when it is commissioned.
When power is short, new chips and robots are commissioned in the same proportion.
BASE holds the calibration and SCEN the scenarios; flags starting legacy_ restore the rules of the first release.
"""
import math
import numpy as np
from scipy.integrate import solve_ivp

BASE = dict(
    Y0=125e12, YA0=0.05e12, K0=20.3e6, F0=18e6, R0=5.08e6, P0=22.0,
    vR=25_000.0, pC0=37_000.0, gp=math.log(1.37), pF=18_000.0, pR=100_000.0, pP=3e9,
    dK=0.20, dF=0.10, dR=0.08,
    s0=0.008, sM=0.30, sK=0.50, sF=0.20, sR=0.15, sP=0.15,
    w0=22.0 / 20.3, gw=math.log(1.4), wR=1.0, A0=30.0, gA=math.log(1.4), Cref=100e9, c0=0.1, phi=0.6,
    ind0=0.655e6, ind_g=math.log(806/655)/3,
    chip_rule="deliver", old_power=False, gE2=None, tE=5.0,
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
SCEN["trend_only"]       = {**B, "Ta": None}                       # no self-replication
SCEN["replication_only"] = {**B, "LR": 0.0, "psi": 0.0}            # a(t) on, no learning, robots not amplified
SCEN["repl_plus_intel"]  = {**B, "LR": 0.0}                        # + intelligence amplification (psi)
SCEN["repl_plus_learn"]  = {**B, "psi": 0.0}                       # + production learning (LR)
SCEN["human_energy"]     = {**B, "mP": 0.0}
SCEN["absorption_bound"] = {**B, "absorb": math.log(2.0)}
SCEN["displacement"]     = {**B, "disp": 0.5}
SCEN["power_backlash"]   = {**B, "gA": math.log(1.25)}             # datacentre moratoria: human-built power additions +25%/yr
SCEN["software_slow"]    = {**B, "gE2": math.log(1.5)}             # software efficiency 2.5x a year to 2030, 1.5x after
SCEN["software_slow_trend"] = {**B, "gE2": math.log(1.5), "Ta": None}
SCEN["chips_redirect"]   = {**B, "chip_rule": "redirect"}          # chip orders the fabs cannot fill are spent on fabs


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
def robots_human(t, p):
    # Forecasts are calendar-year quantities. The archived calibration treats
    # them as January flow rates; the annual-volume check spreads each quoted
    # quantity evenly over its calendar year before applying power rationing.
    if p.get("annual_robot_volumes"):
        t = math.floor(t)
    return p["ind0"] * math.exp(p["ind_g"] * t) + humanoids(t, p)
def power_use(K, R, t, p):
    """AI power drawn, GW: compute at w0 falling with efficiency, plus robots added since 2025."""
    robots = R if p["old_power"] else max(R - p["R0"], 0.0)
    return K * p["w0"] * math.exp(-p["gw"] * t) * 1e-6 + robots * p["wR"] * 1e-6
def wright(C, p): return (1 + C / p["Cref"]) ** (-math.log2(1 - p["LR"])) if p["LR"] > 0 else 1.0

def outputs(t, y, p):
    K, F, R, P = map(math.exp, y[:4]); C = y[4]          # C integrated in linear space
    E = math.exp(p["gE"] * t) if p["gE2"] is None else math.exp(p["gE"] * min(t, p["tE"]) + p["gE2"] * max(t - p["tE"], 0.0))
    YH = human_gdp(t, p)
    YA = p["YA0"] * ((K / p["K0"]) * E) ** p["rho"]
    YR = p["vR"] * R * E ** p["psi"]
    if p["absorb"] is not None:
        cap = (p["YA0"] + p["vR"] * p["R0"]) * math.exp(p["absorb"] * t)
        if YA + YR > cap:
            f = cap / (YA + YR); YA, YR = YA * f, YR * f
    YH = max(YH - p["disp"] * (YA + YR - p["YA0"] - p["vR"] * p["R0"]), 0.2 * YH)
    return K, F, R, P, C, E, YH, YA, YR, (0.0 if p["Ta"] is None else logistic(t, p["Ta"], p["wa"]))

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
    chips_m = min(I_m_done * p["sK"] / pC, F)                           # machine orders have first claim on fab output
    chips_h = min(I_h * hK / pC, F - chips_m)
    unfilled = I_m_done * p["sK"] - chips_m * pC                        # $ of chip orders the fabs cannot fill
    extra_fabs = unfilled / p["pF"] if p["chip_rule"] == "redirect" else 0.0
    dF = min(I_h * hF / p["pF"], F * (p["gF"] + p["dF"])) + I_m_done * p["sF"] / p["pF"] + extra_fabs - p["dF"] * F
    dK = chips_m + chips_h - p["dK"] * K
    robots_m = I_m_done * p["sR"] / p["pR"]
    gross_robots = robots_human(t, p) + robots_m
    dR = gross_robots - p["dR"] * R
    dP = p["A0"] * math.exp(p["gA"] * t) + p["mP"] * I_m_done * p["sP"] / p["pP"]   # human-built additions path + machine-built
    # Commission chips and robots in equal proportions if their combined draw
    # would exhaust power. Unused resources lapse. Human and machine orders
    # face the same power constraint after the fab allocation.
    chip_watts = p["w0"] * math.exp(-p["gw"] * t) * 1e-6
    robot_watts = p["wR"] * 1e-6
    demand = power_use(K, R, t, p)
    retiring_load = chip_watts * K * (p["gw"] + p["dK"]) + robot_watts * p["dR"] * R
    new_load = chip_watts * (chips_m + chips_h) + robot_watts * gross_robots
    headroom = max(P - demand, 0.0)
    available = dP + retiring_load + headroom / p.get("dispatch_window", 0.01)
    fraction = min(1.0, available / new_load) if new_load else 1.0
    dK = fraction * (chips_m + chips_h) - p["dK"] * K
    dR = fraction * gross_robots - p["dR"] * R
    if p.get("legacy_power_rule"):
        dK = chips_m + chips_h - p["dK"] * K
        dR = gross_robots - p["dR"] * R
        if demand >= 0.999 * P:
            dK = min(dK, K * (dP / P + p["gw"]))
            dR = min(dR, R * dP / P)
    chip_accept = (dK + p["dK"] * K) / (chips_m + chips_h) if chips_m + chips_h else 0.
    robot_accept = (dR + p["dR"] * R) / gross_robots
    machine_fabs = p["sF"] * I_m_done + extra_fabs * p["pF"]
    machine_power = p["mP"] * p["sP"] * I_m_done
    dC = chip_accept * chips_m * pC + robot_accept * robots_m * p["pR"] + machine_fabs + machine_power
    if p.get("legacy_experience"):
        dC = I_m_done - unfilled
    return [dK / K, dF / F, dR / R, dP / P, dC]

def run(name, over=None, t_max=T_MAX):
    p = {**BASE, **SCEN[name], **(over or {})}
    y0 = [math.log(p["K0"]), math.log(p["F0"]), math.log(p["R0"]), math.log(p["P0"]), 0.0]   # C0 = 0
    tau = p["tau"]; chunks = []
    def hist(t):
        if not chunks: return y0
        for (t0, t1, sol) in chunks:
            if t0 - 1e-12 <= t <= t1 + 1e-12: return sol.sol(t)
        raise ValueError(f"requested time {t} is outside solver history")
    def blow(t, y): return p.get("stock_stop", BLOW) - (y[0] - math.log(p["K0"]))
    def fast(t, y): return p.get("rate_stop", RATE_STOP) - rhs(t, y, p, hist)[0]
    blow.terminal = fast.terminal = True
    t, y, t_star, rate_at_stop = 0.0, y0, None, None
    step = tau if tau > 0 else t_max
    while t < t_max - 1e-9:
        t1 = min(t + step, t_max)
        if p.get("annual_robot_volumes"):
            t1 = min(t1, math.floor(t + 1e-10) + 1)
        sol = solve_ivp(lambda tt, yy: rhs(tt, yy, p, hist), (t, t1), y, events=[blow, fast], max_step=p.get("max_step", MAX_STEP),
                        rtol=p.get("rtol", 1e-7), atol=[1e-9] * 4 + [1.0], dense_output=True, method="RK45")   # C is in dollars
        if not sol.success:
            raise RuntimeError(sol.message)
        chunks.append((t, sol.t[-1], sol))
        hit = [(ev[0], i) for i, ev in enumerate(sol.t_events) if len(ev)]
        if hit:
            te, i = min(hit); rate_at_stop = rhs(te, sol.y_events[i][0], p, hist)[0]
            t_star = ("stock", "rate")[i]  # finite numerical event only
            t = te; break
        t, y = sol.t[-1], sol.y[:, -1]
    t_last = t
    def st(tt):
        K, F, R, P, C, E, YH, YA, YR, a = outputs(tt, hist(tt), p)
        return dict(year=2026 + tt, Y=YH + YA + YR, YH=YH, YA=YA, YR=YR, K=K, F=F, R=R, P=P, C=C, a=a, W=wright(C, p),
                    GW=power_use(K, R, tt, p), mshare=(YA + YR) / (YH + YA + YR))
    def growth(tt): return st(tt)["Y"] / st(tt - 1)["Y"] - 1
    g30 = mach = None
    for tt in np.arange(1.0, t_last, 0.01):
        s = st(tt)
        if g30 is None and growth(tt) > 0.30: g30 = 2026 + tt
        if mach is None and s["YA"] + s["YR"] > s["YH"]: mach = 2026 + tt
    return p, st, growth, t_star, t_last, g30, mach

