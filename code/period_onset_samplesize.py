"""Subgroup-by-period interaction tests; treated-state comparison on a time-since-symptom-onset axis; E-value
sensitivity to the slope SD; approximate sample size for untreated genetic PD.
Run: python period_onset_samplesize.py <PPMI_Curated_Data_Cut_Public_20260511.xlsx>   Writes period_onset_samplesize_results.json.
"""
import sys, json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
exec(open("pipeline.py").read())
raw = pd.read_excel(sys.argv[1], sheet_name="20260511")
A, BL = build_cohort(raw)
A["post"] = (pd.to_datetime(A.visit_date) >= pd.Timestamp("2024-12-01")).astype(int)
X = set_groups(A, GROUPS3); X["t2"] = X.t ** 2
FB = "G ~ t*C(sg) + " + COV
OUT = {}
m0 = fit_lmm(FB + " + t2 + post", X); m1 = fit_lmm(FB + " + t2 + post + post:C(sg)", X)
m2 = fit_lmm(FB + " + t2 + post + post:C(sg) + post:t + post:t:C(sg)", X)
OUT["period_by_group"] = dict(chi2=lrt(m1, m0, 2)[0], p=lrt(m1, m0, 2)[1], gba_extra=m1.params["post:C(sg)[T.GBA]"])
OUT["period_by_time_by_group"] = dict(chi2=lrt(m2, m1, 3)[0], p=lrt(m2, m1, 3)[1])
# onset axis
bl = raw[raw.EVENT_ID == "BL"].drop_duplicates("PATNO").set_index("PATNO")
pp = X.drop_duplicates("PATNO").set_index("PATNO")
bg = pp[pp.subgroup != "Sporadic PD"].dropna(subset=["trt0"]); gt = set(bg.index[bg.trt0 == 1])
Z = X.copy(); Z["on"] = (Z.LEDD > 0).astype(int)
Z["ageonset"] = Z.PATNO.map(bl.ageonset); Z["agediag"] = Z.PATNO.map(bl.agediag); Z = Z.dropna(subset=["ageonset"])
Z["tdx"] = Z.dur + Z.t + (Z.agediag - Z.ageonset)
D = pd.concat([Z[(Z.subgroup == "Sporadic PD") & (Z.on == 1)], Z[Z.PATNO.isin(gt) & (Z.on == 1)]])
D = D[D.groupby("PATNO").t.transform("nunique") >= 2].copy()
g_, s_ = D[D.subgroup != "Sporadic PD"].tdx, D[D.subgroup == "Sporadic PD"].tdx
D = D[(D.tdx >= max(g_.quantile(.02), s_.quantile(.02))) & (D.tdx <= min(g_.quantile(.98), s_.quantile(.98)))]
D = D[D.groupby("PATNO").t.transform("nunique") >= 2].copy(); D["gen"] = (D.subgroup != "Sporadic PD").astype(int)
D["tdxc"] = D.tdx - D.tdx.mean(); D["tdx2"] = D.tdxc ** 2
b = D.drop_duplicates("PATNO"); D["agec"] = D.age0 - b.age0.mean(); D["LEDDc"] = (D.LEDD - D.LEDD.mean()) / 100
cov = " + tdx2 + agec + C(sex) + LEDDc"; best = {}
for key, f in (("full", "G ~ tdxc*gen" + cov), ("null", "G ~ tdxc + gen" + cov)):
    for meth in (["powell", "lbfgs"], ["nm", "lbfgs"], ["bfgs"]):
        try:
            r = smf.mixedlm(f, D, groups=D.PATNO, re_formula="~tdxc").fit(reml=False, method=meth)
            if r.converged and (key not in best or r.llf > best[key].llf):
                best[key] = r
        except Exception:
            pass
est, se = -best["full"].params["tdxc:gen"], best["full"].bse["tdxc:gen"]
OUT["on_treatment_onset_axis"] = dict(n=int(D.PATNO.nunique()), diff=est, lo=est - 1.96 * se, hi=est + 1.96 * se, p=lrt(best["full"], best["null"], 1)[1])
# E-value SD sensitivity
sds = {}
for g in GROUPS3:
    m = fit_lmm("G ~ t + agec + C(sex) + durc + agec:t + durc:t", X[X.subgroup == g].copy(), reml=True)
    sds[g] = float(np.sqrt(m.cov_re.iloc[1, 1]))
prim = fit_lmm(FB, X.drop(columns=["t2"]), reml=True); sds["pooled"] = float(np.sqrt(prim.cov_re.iloc[1, 1]))
OUT["slope_sds"] = sds
# sample size for untreated genetic PD (SE scaled by 1/sqrt(n), 80% power, two-sided alpha 0.05)
se_now = (0.00919 + 0.00150) / (2 * 1.96)
OUT["n_untreated_genetic_needed"] = {str(d): float(61 * (se_now / (d / 2.8)) ** 2) for d in (0.0114, 0.0057, 0.00385)}
json.dump(OUT, open("period_onset_samplesize_results.json", "w"), indent=1, default=float)
print("DONE", flush=True)
