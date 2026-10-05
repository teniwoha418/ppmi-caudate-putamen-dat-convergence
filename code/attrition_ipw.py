"""Scan availability and attrition, inverse-probability-of-observation weighted GEE, and residual and
near-floor caudate checks (Supplementary Table S5; Supplementary Methods M8).
Run: python attrition_ipw.py <PPMI_Curated_Data_Cut_Public_20260511.xlsx>   -> attrition_results.json
"""
import sys, json
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
exec(open("pipeline.py").read())

raw = pd.read_excel(sys.argv[1], sheet_name="20260511")
A, BL = build_cohort(raw)
X = set_groups(A, GROUPS3)
FG = "G ~ t*C(sg) + " + COV
ST = ["Sporadic PD (untreated)", "Genetic, untreated", "Genetic, treated"]
OUT = {}

cut = pd.to_datetime(raw.visit_date, format="%m/%Y", errors="coerce").max()
bdate = pd.to_datetime(raw[raw.EVENT_ID == "BL"].drop_duplicates("PATNO").set_index("PATNO").visit_date,
                       format="%m/%Y", errors="coerce")
X["stratum"] = np.where(X.subgroup == "Sporadic PD", "Sporadic PD", np.where(X.trt0 == 1, "Genetic, treated", "Genetic, untreated"))
X["win"] = pd.cut(X.t, [-0.01, 0.0, 0.5, 1.6, 3.0, 4.6, 99], labels=["BL", "early", "W1", "W2", "W3", "late"])
WIN = [("W1", 0.5, 1.6), ("W2", 1.6, 3.0), ("W3", 3.0, 4.6)]

# window-level observation data; a window enters only once it has fully elapsed before the data cut
rows = []
for pid, g in X.groupby("PATNO"):
    b = g.iloc[0]; lastG = g[g.t == 0].G.iloc[0]; prev = 1
    for w, lo, hi in WIN:
        if pd.isna(bdate.get(pid)) or bdate[pid] + pd.DateOffset(months=int(hi * 12)) > cut:
            break
        gw = g[(g.t > lo) & (g.t <= hi)]; obs = int(len(gw) > 0)
        rows.append(dict(PATNO=pid, win=w, obs=obs, prev_obs=prev, lagG=lastG, stratum=b.stratum, age0=b.age0,
                         sex=b.sex, dur=b.dur, G0=b.G0, L0=b.L0, phase=b.phase))
        if obs:
            lastG = gw.G.iloc[-1]
        prev = obs
W = pd.DataFrame(rows)
OUT["observation_by_window"] = (W.groupby(["win", "stratum"]).obs.agg(["size", "mean"]).reset_index().to_dict("records"))

# attrition model for the scheduled 2- and 4-year windows (1-year imaging is protocol-dependent)
W23 = W[W.win.isin(["W2", "W3"])].copy()
lg = smf.logit("obs ~ C(win)*C(stratum) + age0 + C(sex) + dur + G0 + L0 + lagG + prev_obs + C(phase)", W23).fit(disp=0)
lg0 = smf.logit("obs ~ C(win)*C(stratum)", W23).fit(disp=0)
W23["sw"] = lg0.predict(W23) / lg.predict(W23)
cap = W23.loc[W23.obs == 1, "sw"].quantile(0.99); W23["sw"] = W23.sw.clip(upper=cap)
OUT["attrition_model"] = {k: dict(OR=float(np.exp(lg.params[k])), p=float(lg.pvalues[k])) for k in ["G0", "L0", "lagG", "prev_obs", "age0", "dur"]}
OUT["weights"] = dict(mean=float(W23.loc[W23.obs == 1, "sw"].mean()), min=float(W23.loc[W23.obs == 1, "sw"].min()),
                      max=float(W23.loc[W23.obs == 1, "sw"].max()), cap99=float(cap))
wmap = W23[W23.obs == 1].set_index(["PATNO", "win"]).sw
X["w"] = [wmap.get((p, str(w)), 1.0) if str(w) in ("W2", "W3") else 1.0 for p, w in zip(X.PATNO, X.win)]
Xg = X[X.t <= 4.6].copy()


def gee(d, groups, name_col, wcol=None):
    d = d.copy(); d["sg"] = pd.Categorical(d[name_col], categories=groups); b = d.drop_duplicates("PATNO")
    for c, s in [("agec", "age0"), ("durc", "dur")]:
        d[c] = d[s] - b[s].mean()
    kw = dict(weights=d[wcol].values) if wcol else {}
    r = smf.gee(FG, "PATNO", d, cov_struct=sm.cov_struct.Independence(), family=sm.families.Gaussian(), **kw).fit()
    names = [f"t:C(sg)[T.{g}]" for g in groups[1:]]
    bb = r.params[names].values; VV = r.cov_params().loc[names, names].values
    out = dict(wald_p=float(stats.chi2.sf(float(bb @ np.linalg.solve(VV, bb)), len(names))))
    for g, n in zip(groups[1:], names):
        out[g] = dict(sporadic_minus=float(-r.params[n]), p=float(r.pvalues[n]))
    if groups == ST:
        diff = r.params[names[0]] - r.params[names[1]]; se = np.sqrt(VV[0, 0] + VV[1, 1] - 2 * VV[0, 1])
        out["untreated_minus_treated_slope"] = dict(est=float(diff), lo=float(diff - 1.96 * se), hi=float(diff + 1.96 * se),
                                                    p=float(2 * stats.norm.sf(abs(diff / se))))
    return out

for lab, wc in (("unweighted", None), ("ipw", "w")):
    OUT[f"gee_3group_{lab}"] = gee(Xg, GROUPS3, "subgroup", wc)
Qg = Xg.dropna(subset=["trt0"]); Qg = Qg[~((Qg.subgroup == "Sporadic PD") & (Qg.trt0 == 1))].copy()
Qg["sname"] = Qg.stratum.replace({"Sporadic PD": ST[0]})
for lab, wc in (("unweighted", None), ("ipw", "w")):
    OUT[f"gee_strata_{lab}"] = gee(Qg, ST, "sname", wc)

# Residuals of the primary REML model and exclusion of near-floor caudate scans
r = fit_lmm(FG, X, reml=True)
OUT["residuals"] = dict(skew=float(stats.skew(r.resid)), excess_kurtosis=float(stats.kurtosis(r.resid)))
Xc = X[X.MIA_CAUDATE_BILAT >= 0.1]; Xc = Xc[Xc.groupby("PATNO").t.transform("nunique") >= 2]
Xc = set_groups(Xc, GROUPS3)
full, null = fit_lmm(FG, Xc), fit_lmm("G ~ t + C(sg) + " + COV, Xc)
tk = tukey_contrasts(fit_lmm(FG, Xc, reml=True), GROUPS3)
OUT["exclude_caudate_lt_0.1"] = dict(scans_excluded=int(len(X) - len(Xc)), n=int(Xc.PATNO.nunique()), p=float(lrt(full, null, 2)[1]),
                                     gba=float(tk.loc[0, "diff"]), lrrk2=float(tk.loc[1, "diff"]))
json.dump(OUT, open("attrition_results.json", "w"), indent=1, default=float)
print("DONE", flush=True)
