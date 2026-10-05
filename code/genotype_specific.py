"""Genotype-specific treated-state comparison and within-genotype duration-matched pairs (Supplementary Table S13, section F).
Run: python genotype_specific.py <PPMI_Curated_Data_Cut_Public_20260511.xlsx>   Writes genotype_specific_results.json.
"""
import sys, json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
exec(open("pipeline.py").read())
raw = pd.read_excel(sys.argv[1], sheet_name="20260511")
A, BL = build_cohort(raw)
X = set_groups(A, GROUPS3)
pp = X.drop_duplicates("PATNO").set_index("PATNO")
bg = pp[pp.subgroup != "Sporadic PD"].dropna(subset=["trt0"])
Z = X.copy(); Z["tdx"] = Z.dur + Z.t; Z["on"] = (Z.LEDD > 0).astype(int)
OUT = {}

def on_cmp(gen_ids):
    D = pd.concat([Z[(Z.subgroup == "Sporadic PD") & (Z.on == 1)], Z[Z.PATNO.isin(gen_ids) & (Z.on == 1)]])
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
    e, se = -best["full"].params["tdxc:gen"], best["full"].bse["tdxc:gen"]
    return dict(n_gen=int(b.gen.sum()), diff=e, lo=e - 1.96 * se, hi=e + 1.96 * se, p=lrt(best["full"], best["null"], 1)[1])

def matched(g, adj_u):
    b_ = bg[bg.subgroup == g]; du, dt = b_[b_.trt0 == 0].dur, b_[b_.trt0 == 1].dur; tre, prs = dt.copy(), []
    for pid, d0 in du.sort_values().items():
        if tre.empty:
            break
        j = (tre - d0).abs().idxmin()
        if abs(tre[j] - d0) <= 0.25:
            prs.append((pid, j)); tre = tre.drop(j)
    ids = {a for a, b in prs} | {b for a, b in prs}
    d = X[X.PATNO.isin(ids)].copy(); d["u0"] = d.PATNO.map(pp.updrs3_score)
    if adj_u:
        d = d.dropna(subset=["u0"])
    b2 = d.drop_duplicates("PATNO")
    for c, s in [("agec", "age0"), ("durc", "dur"), ("u0c", "u0")]:
        d[c] = d[s] - b2[s].mean()
    extra = " + u0c + u0c:t" if adj_u else ""
    m1 = fit_lmm("G ~ t*trt0 + " + COV + extra, d); m0 = fit_lmm("G ~ t + trt0 + " + COV + extra, d)
    e, se = -m1.params["t:trt0"], m1.bse["t:trt0"]
    return dict(pairs=len(prs), n=int(d.PATNO.nunique()), diff=e, lo=e - 1.96 * se, hi=e + 1.96 * se, p=lrt(m1, m0, 1)[1])

for g in ["GBA", "LRRK2"]:
    OUT[f"on_treatment_{g}"] = on_cmp(set(bg.index[(bg.trt0 == 1) & (bg.subgroup == g)]))
    OUT[f"matched_{g}"] = matched(g, False); OUT[f"matched_updrs_{g}"] = matched(g, True)
json.dump(OUT, open("genotype_specific_results.json", "w"), indent=1, default=float)
print("DONE", flush=True)
