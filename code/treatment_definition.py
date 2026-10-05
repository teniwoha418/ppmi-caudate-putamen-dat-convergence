"""Treatment-definition sensitivity: PDTRTMNT (MDS-UPDRS III form item: dopaminergic medication or DBS) vs LEDD (medication log).
Counts discordance and reruns the treatment-dependent analyses with 'treated' = PDTRTMNT==1 OR LEDD>0 (baseline and scan level).
Run: python treatment_definition.py <PPMI_Curated_Data_Cut_Public_20260511.xlsx>   Writes treatment_definition_results.json.
"""
import sys, json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
exec(open("pipeline.py").read())
raw = pd.read_excel(sys.argv[1], sheet_name="20260511")
A, BL = build_cohort(raw)
X = set_groups(A, GROUPS3).sort_values(["PATNO", "t"])
X["gen"] = (X.subgroup != "Sporadic PD").astype(int)
first = X.drop_duplicates("PATNO").set_index("PATNO")
OUT = {}
# --- counts
X["onL"] = (X.LEDD > 0).astype(int)
OUT["baseline_crosstab"] = {f"{'genetic' if g else 'sporadic'}|PDTRTMNT={int(p)}|LEDD>0={int(l)}": int(n)
    for (g, p, l), n in first.assign(l=(first.LEDD > 0)).groupby(["gen", "trt0", "l"]).size().items()}
OUT["scan_discordant"] = dict(total_scans=int(len(X)),
    pdtrt0_ledd_pos=int(((X.PDTRTMNT == 0) & (X.onL == 1)).sum()), pdtrt0_ledd_pos_participants=int(X[(X.PDTRTMNT == 0) & (X.onL == 1)].PATNO.nunique()),
    pdtrt1_ledd0=int(((X.PDTRTMNT == 1) & (X.onL == 0)).sum()), pdtrt1_ledd0_participants=int(X[(X.PDTRTMNT == 1) & (X.onL == 0)].PATNO.nunique()),
    share_ledd100_among_pdtrt0_pos=float((X[(X.PDTRTMNT == 0) & (X.onL == 1)].LEDD == 100).mean()))
# --- alternative definitions
X["trtA"] = X.PATNO.map(((first.trt0 == 1) | (first.LEDD > 0)).astype(int))
X["onA"] = ((X.LEDD > 0) | (X.PDTRTMNT == 1)).astype(int)
FG = "G ~ t*C(sg) + " + COV; FG0 = "G ~ t + C(sg) + " + COV

def primary(D, groups):
    full, null = fit_lmm(FG, D), fit_lmm(FG0, D)
    tk = tukey_contrasts(fit_lmm(FG, D, reml=True), groups)
    return dict(n=int(D.PATNO.nunique()), p=lrt(full, null, 2)[1], tukey=tk.to_dict("records"))

for lab, col in [("PDTRTMNT", "trt0"), ("either", "trtA")]:
    U = set_groups(X[X[col] == 0], GROUPS3); OUT[f"untreated_{lab}"] = primary(U, GROUPS3)
    Q = X.dropna(subset=[col]).copy(); Q = Q[~((Q.gen == 0) & (Q[col] == 1))]
    ST = ["Sporadic PD (untreated)", "Genetic, untreated", "Genetic, treated"]
    Q["sg"] = pd.Categorical(np.where(Q.gen == 0, ST[0], np.where(Q[col] == 1, ST[2], ST[1])), categories=ST)
    b = Q.drop_duplicates("PATNO")
    for c, s in [("agec", "age0"), ("durc", "dur"), ("L0c", "L0")]:
        Q[c] = Q[s] - b[s].mean()
    OUT[f"strata_{lab}"] = primary(Q, ST)
    OUT[f"strata_{lab}"]["n_by_stratum"] = b.sg.value_counts().to_dict()

# duration-matched pairs within genetic PD
def matched(col, adj_u=False):
    bg = first[first.gen == 1]
    du, dt = bg[bg[col if col != "trtA" else "trt0"].isna() == False], None
    bb = bg.assign(tr=((bg.trt0 == 1) | (bg.LEDD > 0)).astype(int) if col == "trtA" else bg.trt0)
    du, dt = bb[bb.tr == 0].dur, bb[bb.tr == 1].dur; tre, prs = dt.copy(), []
    for pid, d0 in du.sort_values().items():
        if tre.empty:
            break
        j = (tre - d0).abs().idxmin()
        if abs(tre[j] - d0) <= 0.25:
            prs.append((pid, j)); tre = tre.drop(j)
    ids = {a for a, b_ in prs} | {b_ for a, b_ in prs}
    D = X[X.PATNO.isin(ids)].copy(); D["tr"] = D.PATNO.map(bb.tr); D["sg"] = pd.Categorical(D.subgroup, categories=["GBA", "LRRK2"])
    D["u0"] = D.PATNO.map(first.updrs3_score)
    if adj_u:
        D = D.dropna(subset=["u0"])
    b2 = D.drop_duplicates("PATNO")
    for c, s in [("agec", "age0"), ("durc", "dur"), ("u0c", "u0")]:
        D[c] = D[s] - b2[s].mean()
    ex = " + u0c + u0c:t" if adj_u else ""
    m1 = fit_lmm("G ~ t*C(sg) + t*tr + " + COV + ex, D); m0 = fit_lmm("G ~ t*C(sg) + tr + " + COV + ex, D)
    e, se = -m1.params["t:tr"], m1.bse["t:tr"]
    return dict(pairs=len(prs), n=int(D.PATNO.nunique()), diff=e, lo=e - 1.96 * se, hi=e + 1.96 * se, p=lrt(m1, m0, 1)[1])

for lab in ["trt0", "trtA"]:
    OUT[f"matched_{lab}"] = matched(lab); OUT[f"matched_updrs_{lab}"] = matched(lab, True)

# treated-state comparison on disease-time axis
def on_cmp(gen_treated_col, on_col, state=1):
    Z = X.copy(); Z["tdx"] = Z.dur + Z.t; Z["on"] = Z[on_col]
    treated = ((first.trt0 == 1) | (first.LEDD > 0)) if gen_treated_col == "trtA" else (first.trt0 == 1)
    gids = set(first.index[(first.gen == 1) & (treated == (state == 1))])
    D = pd.concat([Z[(Z.gen == 0) & (Z.on == state)], Z[Z.PATNO.isin(gids) & (Z.on == state)]])
    D = D[D.groupby("PATNO").t.transform("nunique") >= 2].copy()
    g_, s_ = D[D.gen == 1].tdx, D[D.gen == 0].tdx
    D = D[(D.tdx >= max(g_.quantile(.02), s_.quantile(.02))) & (D.tdx <= min(g_.quantile(.98), s_.quantile(.98)))]
    D = D[D.groupby("PATNO").t.transform("nunique") >= 2].copy()
    D["tdxc"] = D.tdx - D.tdx.mean(); D["tdx2"] = D.tdxc ** 2
    b = D.drop_duplicates("PATNO"); D["agec"] = D.age0 - b.age0.mean(); D["LEDDc"] = (D.LEDD - D.LEDD.mean()) / 100
    cov = " + tdx2 + agec + C(sex)" + (" + LEDDc" if state == 1 else ""); best = {}
    for key, f in (("full", "G ~ tdxc*gen" + cov), ("null", "G ~ tdxc + gen" + cov)):
        for meth in (["powell", "lbfgs"], ["nm", "lbfgs"], ["bfgs"]):
            try:
                r = smf.mixedlm(f, D, groups=D.PATNO, re_formula="~tdxc").fit(reml=False, method=meth)
                if r.converged and (key not in best or r.llf > best[key].llf):
                    best[key] = r
            except Exception:
                pass
    e, se = -best["full"].params["tdxc:gen"], best["full"].bse["tdxc:gen"]
    return dict(n_sp=int((b.gen == 0).sum()), n_gen=int(b.gen.sum()), diff=e, lo=e - 1.96 * se, hi=e + 1.96 * se, p=lrt(best["full"], best["null"], 1)[1])

OUT["on_treatment_PDTRTMNT_LEDD"] = on_cmp("trt0", "onL")
OUT["on_treatment_either"] = on_cmp("trtA", "onA")
OUT["off_treatment_PDTRTMNT_LEDD"] = on_cmp("trt0", "onL", 0)
OUT["off_treatment_either"] = on_cmp("trtA", "onA", 0)
json.dump(OUT, open("treatment_definition_results.json", "w"), indent=1, default=float)
print("DONE", flush=True)
