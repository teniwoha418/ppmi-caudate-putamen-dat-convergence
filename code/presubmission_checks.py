"""Pre-submission checks (values the manuscript reports but no other script prints).
(1) Mean diagnosis-to-enrollment duration of the untreated and treated members of the 46 duration-matched genetic pairs.
(2) Mean and median baseline MDS-UPDRS III by treatment group in the matched pairs with the score (n=81).
(3) Mean and median earliest on-treatment MDS-UPDRS III, genetic vs sporadic (523 participants).
(4) Region-specific log-scale slopes: log(1+C) and log(1+P) by subgroup with Tukey contrasts (REML), and the share of
    the Gradient contrast attributable to each region (G = log(1+C) - log(1+P), so the decomposition is additive up to
    the small differences in estimated variance components between the separate fits).
(5) Exact likelihood-ratio p for baseline Gradient x time on the cognitive composite, read from
    primary_treatment_cognition_results.json if present.
Run from code/: python presubmission_checks.py <PPMI_Curated_Data_Cut_Public_20260511.xlsx>
Writes presubmission_checks_results.json and prints a summary.
"""
import sys, json, os
import numpy as np, pandas as pd
exec(open("pipeline.py").read())

raw = pd.read_excel(sys.argv[1], sheet_name="20260511")
A, BL = build_cohort(raw)
X = set_groups(A, GROUPS3)
pp = X.drop_duplicates("PATNO").set_index("PATNO")
bg = pp[pp.subgroup != "Sporadic PD"].dropna(subset=["trt0"])
OUT = {}

# (1)-(2) duration matching, identical to design_based_comparisons.py and severity_timing_and_matching.py
du, dt = bg[bg.trt0 == 0].dur, bg[bg.trt0 == 1].dur
tre, prs = dt.copy(), []
for pid, d0 in du.sort_values().items():
    if tre.empty:
        break
    j = (tre - d0).abs().idxmin()
    if abs(tre[j] - d0) <= 0.25:
        prs.append((pid, j)); tre = tre.drop(j)
unt_ids, trt_ids = [a for a, b in prs], [b for a, b in prs]
OUT["matched_pairs"] = len(prs)
OUT["matched_mean_duration"] = dict(untreated=float(du.loc[unt_ids].mean()), treated=float(dt.loc[trt_ids].mean()))
u0 = pd.Series(BL.updrs3_score).reindex(unt_ids + trt_ids)
grp = pd.Series(["untreated"] * len(unt_ids) + ["treated"] * len(trt_ids), index=unt_ids + trt_ids)
m = pd.DataFrame(dict(u0=u0, grp=grp)).dropna()
OUT["matched_updrs"] = dict(n=int(len(m)), **{g: dict(mean=float(s.mean()), median=float(s.median()), n=int(len(s)))
                                              for g, s in m.groupby("grp").u0})

# (3) earliest on-treatment MDS-UPDRS III, identical data preparation to severity_timing_and_matching.py
Z = X.copy(); Z["tdx"] = Z.dur + Z.t; Z["on"] = (Z.LEDD > 0).astype(int)
gt = set(bg.index[bg.trt0 == 1])
D = pd.concat([Z[(Z.subgroup == "Sporadic PD") & (Z.on == 1)], Z[Z.PATNO.isin(gt) & (Z.on == 1)]])
D = D[D.groupby("PATNO").t.transform("nunique") >= 2].copy()
g_, s_ = D[D.subgroup != "Sporadic PD"].tdx, D[D.subgroup == "Sporadic PD"].tdx
D = D[(D.tdx >= max(g_.quantile(.02), s_.quantile(.02))) & (D.tdx <= min(g_.quantile(.98), s_.quantile(.98)))]
D = D[D.groupby("PATNO").t.transform("nunique") >= 2].copy(); D["gen"] = (D.subgroup != "Sporadic PD").astype(int)
earliest = D.sort_values(["PATNO", "t"]).groupby("PATNO").first()
e = earliest.dropna(subset=["updrs3_score"])
OUT["earliest_on_treatment_updrs"] = dict(n=int(len(e)), **{("genetic" if g == 1 else "sporadic"): dict(mean=float(s.mean()), median=float(s.median()), n=int(len(s)))
                                                             for g, s in e.groupby("gen").updrs3_score})

# (4) region-specific log-scale contrasts
X["lC"] = np.log(1 + X.MIA_CAUDATE_BILAT); X["lP"] = np.log(1 + X.MIA_PUTAMEN_BILAT)
REG = {}
for y in ["G", "lC", "lP"]:
    f, f0 = f"{y} ~ t*C(sg) + " + COV, f"{y} ~ t + C(sg) + " + COV
    s, p = lrt(fit_lmm(f, X), fit_lmm(f0, X), 2)
    r = fit_lmm(f, X, reml=True)
    REG[y] = dict(chi2=float(s), p=float(p), slopes=group_slopes(r, GROUPS3).to_dict("records"),
                  tukey=tukey_contrasts(r, GROUPS3).to_dict("records"))
OUT["regional_log_scale"] = REG
share = {}
for k, g in [(0, "GBA"), (1, "LRRK2")]:
    # tukey diff = slope(sporadic) - slope(group): positive = faster decline of the Gradient in the genetic group
    dC = REG["lC"]["tukey"][k]["diff"]; dP = REG["lP"]["tukey"][k]["diff"]; dG = REG["G"]["tukey"][k]["diff"]
    caud, put = dC, -dP   # caudate: faster caudate decline in the group; putamen: slower putaminal decline in the group
    share[g] = dict(gradient_contrast=dG, caudate_component=caud, putamen_component=put, sum_of_components=caud + put,
                    caudate_share=caud / (caud + put), putamen_share=put / (caud + put),
                    caudate_contrast_ci=(REG["lC"]["tukey"][k]["lo"], REG["lC"]["tukey"][k]["hi"], REG["lC"]["tukey"][k]["p_tukey"]),
                    putamen_contrast_ci=(REG["lP"]["tukey"][k]["lo"], REG["lP"]["tukey"][k]["hi"], REG["lP"]["tukey"][k]["p_tukey"]))
OUT["regional_share"] = share

# (5) exact cognition p
if os.path.exists("primary_treatment_cognition_results.json"):
    cg = json.load(open("primary_treatment_cognition_results.json"))["cognition"]
    OUT["cognition_G0z_t_lrt_p"] = cg["G0z_t"][3]

json.dump(OUT, open("presubmission_checks_results.json", "w"), indent=1, default=float)
print("matched pairs:", OUT["matched_pairs"], "mean duration untreated/treated:",
      round(OUT["matched_mean_duration"]["untreated"], 2), round(OUT["matched_mean_duration"]["treated"], 2))
print("matched MDS-UPDRS III:", OUT["matched_updrs"])
print("earliest on-treatment MDS-UPDRS III:", OUT["earliest_on_treatment_updrs"])
for g, v in share.items():
    print(g, "Gradient contrast %.5f = caudate %.5f + putamen %.5f (caudate share %.0f%%)" %
          (v["gradient_contrast"], v["caudate_component"], v["putamen_component"], 100 * v["caudate_share"]))
if "cognition_G0z_t_lrt_p" in OUT:
    print("cognition Gradient x time LRT p:", OUT["cognition_G0z_t_lrt_p"])
