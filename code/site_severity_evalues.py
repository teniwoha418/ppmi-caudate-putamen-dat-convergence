"""(1) Within-site model in multi-subgroup sites with site and site-by-time fixed effects; (2) on-treatment comparison
adjusted for / matched on the earliest available on-treatment MDS-UPDRS III; (3) E-values (exploratory);
(4) within-participant core-laboratory period step in PD and the PPMI prodromal cohort
(Supplementary Table S13; Supplementary Methods M9-M10).
Run: python site_severity_evalues.py <PPMI_Curated_Data_Cut_Public_20260511.xlsx>   Writes site_severity_evalues_results.json.
"""
import sys, json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
exec(open("pipeline.py").read())
raw = pd.read_excel(sys.argv[1], sheet_name="20260511")
A, BL = build_cohort(raw)
A["post"] = (pd.to_datetime(A.visit_date) >= pd.Timestamp("2024-12-01")).astype(int)
X = set_groups(A, GROUPS3)
FB = "G ~ t*C(sg) + " + COV; FN = "G ~ t + C(sg) + " + COV
OUT = {}
# (1) within-site model
ns = X.groupby("SITE").subgroup.nunique(); XS = set_groups(X[X.SITE.isin(ns.index[ns >= 2])].copy(), GROUPS3)
XS["SITE"] = XS.SITE.astype(str)
f1, f0 = fit_lmm(FB + " + C(SITE) + C(SITE):t", XS), fit_lmm(FN + " + C(SITE) + C(SITE):t", XS)
OUT["within_site"] = dict(sites=int((ns >= 2).sum()), n=int(XS.PATNO.nunique()), chi2=lrt(f1, f0, 2)[0], p=lrt(f1, f0, 2)[1],
                          tukey=tukey_contrasts(fit_lmm(FB + " + C(SITE) + C(SITE):t", XS, reml=True), GROUPS3).to_dict("records"))
# (2) on-treatment comparison with concurrent motor severity
pp = X.drop_duplicates("PATNO").set_index("PATNO")
bg = pp[pp.subgroup != "Sporadic PD"].dropna(subset=["trt0"]); gt = set(bg.index[bg.trt0 == 1])
Z = X.copy(); Z["tdx"] = Z.dur + Z.t; Z["on"] = (Z.LEDD > 0).astype(int)
sp = Z[(Z.subgroup == "Sporadic PD") & (Z.on == 1)]
D = pd.concat([sp, Z[Z.PATNO.isin(gt) & (Z.on == 1)]]); D = D[D.groupby("PATNO").t.transform("nunique") >= 2].copy()
g_, s_ = D[D.subgroup != "Sporadic PD"].tdx, D[D.subgroup == "Sporadic PD"].tdx
D = D[(D.tdx >= max(g_.quantile(.02), s_.quantile(.02))) & (D.tdx <= min(g_.quantile(.98), s_.quantile(.98)))]
D = D[D.groupby("PATNO").t.transform("nunique") >= 2].copy(); D["gen"] = (D.subgroup != "Sporadic PD").astype(int)
first = D.sort_values(["PATNO", "t"]).groupby("PATNO").first()
D["u1"] = D.PATNO.map(first.updrs3_score); D["tdx1"] = D.PATNO.map(first.tdx)

def fit_on(d, extra=""):
    d = d.copy(); d["tdxc"] = d.tdx - d.tdx.mean(); d["tdx2"] = d.tdxc ** 2
    b = d.drop_duplicates("PATNO"); d["agec"] = d.age0 - b.age0.mean(); d["LEDDc"] = (d.LEDD - d.LEDD.mean()) / 100
    d["u1c"] = d.u1 - b.u1.mean()
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
    return dict(n_sp=int((d.drop_duplicates("PATNO").gen == 0).sum()), n_gen=int((d.drop_duplicates("PATNO").gen == 1).sum()),
                diff=est, lo=est - 1.96 * se, hi=est + 1.96 * se, p=lrt(best["full"], best["null"], 1)[1])

OUT["on_treatment"] = fit_on(D)
Du = D.dropna(subset=["u1"])
OUT["on_treatment_updrs_first"] = fit_on(Du, " + u1c + u1c:tdxc")
F = Du.drop_duplicates("PATNO").set_index("PATNO")[["gen", "u1", "tdx1", "age0"]].copy()
zc = []
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
ids = {a for a, b in pairs} | {b for a, b in pairs}
OUT["on_treatment_updrs_matched"] = fit_on(Du[Du.PATNO.isin(ids)])
# (3) E-values: standardize by SD of participant-specific slopes in the primary REML model
prim = fit_lmm(FB, X, reml=True); sd = float(np.sqrt(prim.cov_re.iloc[1, 1])); tk = tukey_contrasts(prim, GROUPS3)

def ev(x):
    if x <= 0:
        return 1.0
    rr = np.exp(0.91 * x / sd); return rr + np.sqrt(rr * (rr - 1))

OUT["slope_sd"] = sd
OUT["evalues"] = {k: (ev(e), ev(l)) for k, e, l in [("gba", tk.loc[0, "diff"], tk.loc[0, "lo"]), ("lrrk2", tk.loc[1, "diff"], tk.loc[1, "lo"]),
                  ("on_treatment", OUT["on_treatment"]["diff"], OUT["on_treatment"]["lo"]),
                  ("on_treatment_updrs_matched", OUT["on_treatment_updrs_matched"]["diff"], OUT["on_treatment_updrs_matched"]["lo"])]}
# (4) within-participant period step
Xq = X.copy(); Xq["t2"] = Xq.t ** 2
both = Xq.groupby("PATNO").post.nunique(); Pb = Xq[Xq.PATNO.isin(both.index[both == 2])]
OUT["pd_both_period_step"] = {y: fit_lmm(f"{y} ~ t + t2 + agec + C(sex) + durc + post", Pb.assign(lC=np.log1p(Pb.MIA_CAUDATE_BILAT), lP=np.log1p(Pb.MIA_PUTAMEN_BILAT))).params["post"] for y in ["G", "lC", "lP"]}
pr = raw[raw.COHORT == 4].dropna(subset=["MIA_CAUDATE_BILAT", "MIA_PUTAMEN_BILAT", "age_at_visit"]).copy()
pr["post"] = (pd.to_datetime(pr.visit_date) >= pd.Timestamp("2024-12-01")).astype(int)
pr["G"] = np.log((1 + pr.MIA_CAUDATE_BILAT) / (1 + pr.MIA_PUTAMEN_BILAT)); pr["lC"] = np.log1p(pr.MIA_CAUDATE_BILAT); pr["lP"] = np.log1p(pr.MIA_PUTAMEN_BILAT)
pr = pr.sort_values(["PATNO", "age_at_visit"]); pr["t"] = pr.age_at_visit - pr.groupby("PATNO").age_at_visit.transform("min"); pr["t2"] = pr.t ** 2
pr["agec"] = pr.groupby("PATNO").age_at_visit.transform("min"); pr["agec"] -= pr.agec.mean()
OUT["prodromal_period_step"] = {y: fit_lmm(f"{y} ~ t + t2 + agec + post", pr).params["post"] for y in ["G", "lC", "lP"]}
json.dump(OUT, open("site_severity_evalues_results.json", "w"), indent=1, default=float)
print("DONE", flush=True)
