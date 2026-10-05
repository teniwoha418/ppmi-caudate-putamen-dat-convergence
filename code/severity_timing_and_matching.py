"""(1) MDS-UPDRS III adjustment in duration-matched genetic pairs estimated in the same participants with and without
adjustment; (2) on-treatment comparison with the earliest available on-treatment MDS-UPDRS III (unadjusted and adjusted
in the same participants), the score at the first included on-treatment scan, the timing of later scores, and the
structure and balance (standardized mean differences) of the up-to-1:3 matched sample (Table 2; Supplementary Table S13).
Run: python severity_timing_and_matching.py <PPMI_Curated_Data_Cut_Public_20260511.xlsx>
Writes severity_timing_and_matching_results.json.
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
OUT = {}
# (1) pooled 1:1 duration-matched genetic pairs (caliper 0.25 y), same participants with/without MDS-UPDRS III
du, dt = bg[bg.trt0 == 0].dur, bg[bg.trt0 == 1].dur; tre, prs = dt.copy(), []
for pid, d0 in du.sort_values().items():
    if tre.empty:
        break
    j = (tre - d0).abs().idxmin()
    if abs(tre[j] - d0) <= 0.25:
        prs.append((pid, j)); tre = tre.drop(j)
ids = {a for a, b in prs} | {b for a, b in prs}

def within_gen(d, adj_u):
    d = d.copy(); b2 = d.drop_duplicates("PATNO")
    for c, s in [("agec", "age0"), ("durc", "dur"), ("u0c", "u0")]:
        d[c] = d[s] - b2[s].mean()
    extra = " + u0c + u0c:t" if adj_u else ""
    m1 = fit_lmm("G ~ t*C(sg) + t*trt0 + " + COV + extra, d); m0 = fit_lmm("G ~ t*C(sg) + trt0 + " + COV + extra, d)
    e, se = -m1.params["t:trt0"], m1.bse["t:trt0"]
    return dict(n=int(d.PATNO.nunique()), diff=e, lo=e - 1.96 * se, hi=e + 1.96 * se, p=lrt(m1, m0, 1)[1])

d = set_groups(X[X.PATNO.isin(ids)].copy(), ["GBA", "LRRK2"]); d["u0"] = d.PATNO.map(BL.updrs3_score)
du_ = d.dropna(subset=["u0"])
bu = du_.drop_duplicates("PATNO").set_index("PATNO")
OUT["matched_pairs"] = len(prs)
OUT["pairs_both_with_updrs"] = int(sum((a in bu.index) and (b in bu.index) for a, b in prs))
OUT["matched_all"] = within_gen(d.assign(u0=d.u0.fillna(0)), False)
OUT["matched_updrs_subset_unadjusted"] = within_gen(du_, False)
OUT["matched_updrs_subset_adjusted"] = within_gen(du_, True)
# (2) on-treatment comparison
Z = X.copy(); Z["tdx"] = Z.dur + Z.t; Z["on"] = (Z.LEDD > 0).astype(int)
gt = set(bg.index[bg.trt0 == 1])
D = pd.concat([Z[(Z.subgroup == "Sporadic PD") & (Z.on == 1)], Z[Z.PATNO.isin(gt) & (Z.on == 1)]])
D = D[D.groupby("PATNO").t.transform("nunique") >= 2].copy()
g_, s_ = D[D.subgroup != "Sporadic PD"].tdx, D[D.subgroup == "Sporadic PD"].tdx
D = D[(D.tdx >= max(g_.quantile(.02), s_.quantile(.02))) & (D.tdx <= min(g_.quantile(.98), s_.quantile(.98)))]
D = D[D.groupby("PATNO").t.transform("nunique") >= 2].copy(); D["gen"] = (D.subgroup != "Sporadic PD").astype(int)
Ds_ = D.sort_values(["PATNO", "t"])
earliest = Ds_.groupby("PATNO").first()             # first non-missing value per column = earliest available score
strict = Ds_.drop_duplicates("PATNO").set_index("PATNO")  # value recorded at the first included on-treatment scan
# NOTE: D keeps the original (unsorted) row order used in site_severity_evalues.py. The greedy matching below processes genetic
# participants in order of MDS-UPDRS III with an unstable sort, so ties are broken by row order; sorting D first gives
# a slightly different matched sample (157 instead of 156 sporadic controls; estimate 0.02257 instead of 0.02254).
D["u1"] = D.PATNO.map(earliest.updrs3_score); D["u1s"] = D.PATNO.map(strict.updrs3_score); D["tdx1"] = D.PATNO.map(strict.tdx)

def fit_on(d, ucol=None):
    d = d.copy(); d["tdxc"] = d.tdx - d.tdx.mean(); d["tdx2"] = d.tdxc ** 2
    b = d.drop_duplicates("PATNO"); d["agec"] = d.age0 - b.age0.mean(); d["LEDDc"] = (d.LEDD - d.LEDD.mean()) / 100
    extra = ""
    if ucol:
        d["uc"] = d[ucol] - b[ucol].mean(); extra = " + uc + uc:tdxc"
    cov = " + tdx2 + agec + C(sex) + LEDDc" + extra; best = {}
    for key, f in (("full", "G ~ tdxc*gen" + cov), ("null", "G ~ tdxc + gen" + cov)):
        for meth in (["powell", "lbfgs"], ["nm", "lbfgs"], ["bfgs"]):
            try:
                r = smf.mixedlm(f, d, groups=d.PATNO, re_formula="~tdxc").fit(reml=False, method=meth)
                if r.converged and (key not in best or r.llf > best[key].llf):
                    best[key] = r
            except Exception:
                pass
    m = best["full"]; est, se = -m.params["tdxc:gen"], m.bse["tdxc:gen"]
    bb = d.drop_duplicates("PATNO")
    return dict(n_sp=int((bb.gen == 0).sum()), n_gen=int((bb.gen == 1).sum()), diff=est, lo=est - 1.96 * se, hi=est + 1.96 * se,
                p=lrt(best["full"], best["null"], 1)[1], mean_tdx=float(d.tdx.mean()),
                slope_sp=float(m.params["tdxc"]), slope_gen=float(m.params["tdxc"] + m.params["tdxc:gen"]))

OUT["on_treatment"] = fit_on(D)
# time from the first included on-treatment scan to the earliest available on-treatment MDS-UPDRS III
tu = Ds_.dropna(subset=["updrs3_score"]).groupby("PATNO").t.first()
gap = (tu - strict.t.reindex(tu.index)); gap = gap[gap > 1e-9]
OUT["updrs_later_than_first_scan"] = dict(n=int(len(gap)), n_gen=int(strict.gen.reindex(gap.index).sum()) if "gen" in strict else None,
                                          median_years=float(gap.median()), q1=float(gap.quantile(.25)), q3=float(gap.quantile(.75)), max_years=float(gap.max()))
Du = D.dropna(subset=["u1"]); Ds = D.dropna(subset=["u1s"])
OUT["on_earliest_updrs_unadjusted"] = fit_on(Du); OUT["on_earliest_updrs_adjusted"] = fit_on(Du, "u1")
OUT["on_strict_updrs_unadjusted"] = fit_on(Ds); OUT["on_strict_updrs_adjusted"] = fit_on(Ds, "u1s")
F = Du.drop_duplicates("PATNO").set_index("PATNO")[["gen", "u1", "tdx1", "age0"]].copy(); zc = []
for c in ["u1", "tdx1", "age0"]:
    F[c + "z"] = (F[c] - F[c].mean()) / F[c].std(); zc.append(c + "z")
pool, pairs = F[F.gen == 0].copy(), []
for g in F[F.gen == 1].sort_values("u1").index:
    for _ in range(3):
        if pool.empty:
            break
        dist = np.sqrt(((pool[zc] - F.loc[g, zc]) ** 2).sum(axis=1)); j = dist.idxmin()
        if dist[j] <= 0.5:
            pairs.append((g, j)); pool = pool.drop(j)
mids = {a for a, b in pairs} | {b for a, b in pairs}
OUT["match_controls_per_genetic"] = pd.Series([a for a, b in pairs]).value_counts().value_counts().sort_index().to_dict()
def smd(Fm):
    g1, g0 = Fm[Fm.gen == 1], Fm[Fm.gen == 0]
    return {c: float((g1[c].mean() - g0[c].mean()) / np.sqrt((g1[c].var() + g0[c].var()) / 2)) for c in ["u1", "tdx1", "age0"]}
OUT["smd_before"] = smd(F); OUT["smd_after"] = smd(F.loc[sorted(mids)])
OUT["on_updrs_matched"] = fit_on(Du[Du.PATNO.isin(mids)])
json.dump(OUT, open("severity_timing_and_matching_results.json", "w"), indent=1, default=float)
print("DONE", flush=True)
