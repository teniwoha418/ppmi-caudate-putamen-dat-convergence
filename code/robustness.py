"""Three-group (sporadic PD, GBA, LRRK2) Gradient robustness analyses (Table 2; Supplementary Table S2).
Requires cohort_20260511.pkl (see README). Writes robust_3group.csv. Run: python robustness.py <xlsx>
"""
import sys, json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
exec(open("pipeline.py").read())

A = pd.read_pickle("cohort_20260511.pkl")
raw = pd.read_excel(sys.argv[1], sheet_name="20260511")
X = set_groups(A, GROUPS3)
G3 = GROUPS3
rows = []


def record(label, full, null, data, df=2, note="", stat_override=None):
    s, p = lrt(full, null, df) if stat_override is None else stat_override
    tk = tukey_contrasts(full, G3)
    rows.append(dict(analysis=label, participants=data.PATNO.nunique(), scans=len(data),
                     stat=s, df=df, p=p,
                     gba_diff=tk.loc[0, "diff"], gba_p=tk.loc[0, "p_tukey"],
                     lrrk2_diff=tk.loc[1, "diff"], lrrk2_p=tk.loc[1, "p_tukey"],
                     gba_lo=tk.loc[0, "lo"], gba_hi=tk.loc[0, "hi"], lrrk2_lo=tk.loc[1, "lo"], lrrk2_hi=tk.loc[1, "hi"], note=note))
    print(f"{label:45s} n={data.PATNO.nunique()} chi2={s:.2f} p={p:.3g} "
          f"GBA {tk.loc[0,'diff']:.5f} ({tk.loc[0,'p_tukey']:.3g}) "
          f"LRRK2 {tk.loc[1,'diff']:.5f} ({tk.loc[1,'p_tukey']:.3g})", flush=True)


def pair(label, fb, fn, data, re="~t", note=""):
    full = fit_lmm(fb, data, re=re); null = fit_lmm(fn, data, re=re)
    record(label, full, null, data, note=note)
    return full


INT = "t:C(sg)"
FB = "G ~ t*C(sg) + " + COV
FN = "G ~ t + C(sg) + " + COV

# --- primary (ML) for the table row
pair("Primary Gradient", FB, FN, X)

# --- standard supportive sensitivity analyses
d = X[X.t <= 4.5]; d = d[d.groupby("PATNO").t.transform("nunique") >= 2]
pair("Follow-up <=4.5 y", FB, FN, d)
pair("Random intercept only", FB, FN, X, re="1")
d = X[X.EVENT_ID.astype(str).str.match(r"^(BL|V\d+)$")]
d = d[d.groupby("PATNO").t.transform("nunique") >= 2]
pair("Scheduled BL/V visits only", FB, FN, d)
lastdx = (raw[raw.PATNO.isin(set(X.PATNO))].groupby("PATNO").PRIMDIAG
          .apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else 1))
d = X[X.PATNO.map(lastdx).isin([1, 9])]
pair("Last recorded diagnosis remained PD", FB, FN, d)
pair("Raw C-P SBR difference", FB.replace("G ~", "CmP ~"), FN.replace("G ~", "CmP ~"), X)

# --- treatment
d = X.copy()
pair("Time-varying LEDD + LEDD x time", FB + " + LEDDc + LEDDc:t", FN + " + LEDDc + LEDDc:t", d)
d = X.dropna(subset=["trt0"]).copy()
pair("Baseline treatment + treatment x time", FB + " + trt0 + trt0:t", FN + " + trt0 + trt0:t", d)

# --- baseline Gradient (follow-up scans only)
fu = X[X.t > 0].copy()
pair("Follow-up only + baseline Gradient x time", FB + " + G0c + G0c:t", FN + " + G0c + G0c:t", fu)
d = fu.dropna(subset=["trt0"])
pair("Baseline Gradient + baseline treatment", FB + " + G0c + G0c:t + trt0 + trt0:t",
     FN + " + G0c + G0c:t + trt0 + trt0:t", d)
pair("Baseline Gradient + time-varying LEDD", FB + " + G0c + G0c:t + LEDDc + LEDDc:t",
     FN + " + G0c + G0c:t + LEDDc + LEDDc:t", fu)

# --- floor / level (new)
pair("Baseline Level x time (floor test)", FB + " + L0c + L0c:t", FN + " + L0c + L0c:t", X)
pair("Baseline putamen SBR x time (floor test)", FB + " + P0c + P0c:t", FN + " + P0c + P0c:t", X)
d = X[X.t <= 2.6]; d = d[d.groupby("PATNO").t.transform("nunique") >= 2]
pair("Common follow-up window <=2.6 y", FB, FN, d)

# --- design
d = X[X.phase == 1]
pair("Enrollment phase 1 only", FB, FN, d)
d = X[X.dur < 1]
pair("Diagnosis duration <1 y", FB, FN, d)
pair("Sex x time added", FB + " + C(sex):t", FN + " + C(sex):t", X)

# --- nonlinearity
X["t2"] = X.t ** 2
lin = fit_lmm(FB, X); quad = fit_lmm(FB + " + t2", X); quadn = fit_lmm(FN + " + t2", X)
s, p = lrt(quad, lin, 1)
rows.append(dict(analysis="Common quadratic vs linear", participants=X.PATNO.nunique(), scans=len(X),
                 stat=s, df=1, p=p, note="model-comparison test"))
record("Common quadratic time term", quad, quadn, X)
quads = fit_lmm(FB + " + t2 + t2:C(sg)", X)
s, p = lrt(quads, quad, 2)
rows.append(dict(analysis="Subgroup-specific quadratic terms", participants=X.PATNO.nunique(), scans=len(X),
                 stat=s, df=2, p=p, note="model-comparison test"))
print("quadratic tests done", flush=True)

# --- time since diagnosis axis
X["tdx"] = X.dur + X.t; X["tdx2"] = X.tdx ** 2
fb = "G ~ tdx*C(sg) + tdx2 + agec + C(sex)"; fn = "G ~ tdx + C(sg) + tdx2 + agec + C(sex)"
full = fit_lmm(fb, X); null = fit_lmm(fn, X)
tk = tukey_contrasts(full, G3, term_fmt="tdx:C(sg)[T.{}]")
s, p = lrt(full, null, 2)
rows.append(dict(analysis="Time axis: years since diagnosis", participants=X.PATNO.nunique(), scans=len(X),
                 stat=s, df=2, p=p, gba_diff=tk.loc[0, "diff"], gba_p=tk.loc[0, "p_tukey"],
                 lrrk2_diff=tk.loc[1, "diff"], lrrk2_p=tk.loc[1, "p_tukey"]))
print("time-since-diagnosis done", flush=True)

# --- site analyses
X["site"] = X.site.astype(str)
# (a) shared-site LMM with site and site x time fixed effects, pairwise
for g in ["GBA", "LRRK2"]:
    sub = X[X.subgroup.isin(["Sporadic PD", g])].copy()
    shared = set(sub[sub.subgroup == g].site) & set(sub[sub.subgroup == "Sporadic PD"].site)
    sub = sub[sub.site.isin(shared)].copy()
    sub["gi"] = (sub.subgroup == g).astype(int)
    fb = "G ~ t*gi + C(site) + C(site):t + " + COV
    fn = "G ~ t + gi + C(site) + C(site):t + " + COV
    full = fit_lmm(fb, sub); null = fit_lmm(fn, sub)
    s, p = lrt(full, null, 1)
    rows.append(dict(analysis=f"Shared-site LMM (sporadic vs {g})", participants=sub.PATNO.nunique(),
                     scans=len(sub), stat=s, df=1, p=p, note=f"{len(shared)} shared sites; "
                     f"{g}-minus-sporadic = {full.params['t:gi']:.5f}"))
    print("shared site", g, len(shared), s, p, flush=True)
# (b) all-site marginal OLS with site and site x time FE, participant-clustered
ols_f = "G ~ t*C(sg) + C(site) + C(site):t + " + COV
ols = smf.ols(ols_f, X).fit(cov_type="cluster", cov_kwds={"groups": X.PATNO})
names = [f"t:C(sg)[T.{g}]" for g in G3[1:]]
b = ols.params[names].values; V = ols.cov_params().loc[names, names].values
W = float(b @ np.linalg.pinv(V) @ b)
tk = tukey_contrasts(ols, G3)
rows.append(dict(analysis="All-site marginal OLS (site + site x time FE)", participants=X.PATNO.nunique(),
                 scans=len(X), stat=W, df=2, p=stats.chi2.sf(W, 2),
                 gba_diff=tk.loc[0, "diff"], gba_p=tk.loc[0, "p_tukey"],
                 lrrk2_diff=tk.loc[1, "diff"], lrrk2_p=tk.loc[1, "p_tukey"], note="Wald test; exploratory"))
print("all-site OLS done", flush=True)
# (c) random site intercept, participant intercept & slope variance components nested in site
vc = {"pint": "0 + C(PATNO)", "pslope": "0 + C(PATNO):t"}
def fit_site(f):
    best = None
    for meth in (["lbfgs"], ["powell", "lbfgs"]):
        try:
            r = smf.mixedlm(f, X, groups=X.site, re_formula="1", vc_formula=vc).fit(reml=False, method=meth)
            if r.converged and (best is None or r.llf > best.llf): best = r
        except Exception as e:
            print("site fit err", e, flush=True)
    return best
full = fit_site(FB); null = fit_site(FN)
if full is not None and null is not None:
    s, p = lrt(full, null, 2)
    rows.append(dict(analysis="Random site intercept (all sites)", participants=X.PATNO.nunique(),
                     scans=len(X), stat=s, df=2, p=p,
                     gba_diff=-full.params["t:C(sg)[T.GBA]"], gba_p=full.pvalues["t:C(sg)[T.GBA]"],
                     lrrk2_diff=-full.params["t:C(sg)[T.LRRK2]"], lrrk2_p=full.pvalues["t:C(sg)[T.LRRK2]"],
                     gba_lo=-full.conf_int().loc["t:C(sg)[T.GBA]", 1], gba_hi=-full.conf_int().loc["t:C(sg)[T.GBA]", 0],
                     lrrk2_lo=-full.conf_int().loc["t:C(sg)[T.LRRK2]", 1], lrrk2_hi=-full.conf_int().loc["t:C(sg)[T.LRRK2]", 0],
                     note=f"{X.site.nunique()} sites; contrast p unadjusted Wald"))
print("random site done", flush=True)
single = (X.groupby("site").subgroup.nunique() == 1).sum()
rows.append(dict(analysis="Sites with a single subgroup", participants=np.nan, scans=np.nan,
                 note=f"{single} of {X.site.nunique()} sites"))

pd.DataFrame(rows).to_csv("robust_3group.csv", index=False)
print("DONE", flush=True)
