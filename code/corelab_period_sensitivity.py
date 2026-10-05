"""Imaging core-laboratory period sensitivity (the PPMI imaging core laboratory changed from Invicro to XingImaging in
December 2024). Reruns the primary, untreated, treatment-strata, duration-matched and treatment-state-aligned
analyses (a) excluding scans dated from 2024-12-01 onward and (b) with a period step term (Supplementary Table S12).
Run: python corelab_period_sensitivity.py <PPMI_Curated_Data_Cut_Public_20260511.xlsx> <iu_genetic_consensus.csv>
Writes corelab_results.json.
"""
import sys, json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
exec(open("pipeline.py").read())
CUT = pd.Timestamp("2024-12-01")
raw = pd.read_excel(sys.argv[1], sheet_name="20260511")
gc = pd.read_csv(sys.argv[2]).drop_duplicates("PATNO").set_index("PATNO")
A_all, BL = build_cohort(raw)
A_all["post"] = (pd.to_datetime(A_all.visit_date) >= CUT).astype(int)
FG = "G ~ t*C(sg) + " + COV; FG0 = "G ~ t + C(sg) + " + COV
RESULTS = {}
for MODE in ("exclude", "step"):
    OUT = {}
    A = A_all[A_all.post == 0].copy() if MODE == "exclude" else A_all.copy()
    STEP = "" if MODE == "exclude" else " + post"
    FGm, FG0m = FG + STEP, FG0 + STEP
    X3 = set_groups(A, GROUPS3)
    pp = X3.drop_duplicates("PATNO").set_index("PATNO")
    OUT["n_participants"] = int(X3.PATNO.nunique()); OUT["n_scans"] = int(len(X3))

    def primary(X, groups, df):
        full, null = fit_lmm(FGm, X), fit_lmm(FG0m, X)
        s, p = lrt(full, null, df)
        r = fit_lmm(FGm, X, reml=True)
        return dict(n=int(X.PATNO.nunique()), scans=int(len(X)), chi2=s, p=p,
                    slopes=group_slopes(r, groups).to_dict("records"), tukey=tukey_contrasts(r, groups).to_dict("records"))
    OUT["primary_3group"] = primary(X3, GROUPS3, 2)
    U = set_groups(X3[X3.trt0 == 0], GROUPS3)
    OUT["untreated"] = primary(U, GROUPS3, 2)
    Q = X3.dropna(subset=["trt0"]).copy()
    Q = Q[~((Q.subgroup == "Sporadic PD") & (Q.trt0 == 1))]
    Q["stratum"] = np.where(Q.subgroup == "Sporadic PD", "Sporadic PD (untreated)",
                            np.where(Q.trt0 == 1, "Genetic, treated", "Genetic, untreated"))
    ST = ["Sporadic PD (untreated)", "Genetic, untreated", "Genetic, treated"]
    Q["sg"] = pd.Categorical(Q.stratum, categories=ST)
    b = Q.drop_duplicates("PATNO")
    for c, s_ in [("agec", "age0"), ("durc", "dur"), ("L0c", "L0")]:
        Q[c] = Q[s_] - b[s_].mean()
    OUT["strata"] = primary(Q, ST, 2)
    print(MODE, "primary/untreated/strata done", flush=True)

    def three_group(X):
        X = set_groups(X, GROUPS3)
        full, null = fit_lmm(FGm, X), fit_lmm(FG0m, X)
        tk = tukey_contrasts(fit_lmm(FGm, X, reml=True), GROUPS3)
        return dict(n=X.PATNO.nunique(), p=lrt(full, null, 2)[1], gba=tk.loc[0, "diff"], lrrk2=tk.loc[1, "diff"])

    # genetic confirmation
    g = gc.reindex(X3.PATNO.unique())
    spo = set(pp.index[pp.subgroup == "Sporadic PD"])
    var_ids = spo | set(g.index[g.GBA == "N409S"]) | set(g.index[g.LRRK2 == "G2019S"])
    nonc = set(g.index[(g.PATHVAR_COUNT == 0) & (g.GBA == "0") & (g.LRRK2 == "0")])
    OUT["variant_restricted"] = three_group(X3[X3.PATNO.isin(var_ids)])
    OUT["noncarrier_sporadic"] = three_group(X3[X3.PATNO.isin(nonc | (set(X3.PATNO) - spo))])

    # duration matching within genetic PD
    bg = pp[pp.subgroup != "Sporadic PD"].dropna(subset=["trt0"])
    du, dt = bg[bg.trt0 == 0].dur, bg[bg.trt0 == 1].dur
    tre, pairs = dt.copy(), []
    for pid, d0 in du.sort_values().items():
        if tre.empty:
            break
        j = (tre - d0).abs().idxmin()
        if abs(tre[j] - d0) <= 0.25:
            pairs.append((pid, j)); tre = tre.drop(j)
    mids = {a for a, b in pairs} | {b for a, b in pairs}


    def within_gen(ids, extra=""):
        d = X3[X3.PATNO.isin(ids)].copy(); d["sg"] = pd.Categorical(d.subgroup, categories=["GBA", "LRRK2"])
        d["u0"] = d.PATNO.map(BL.updrs3_score)
        if "u0c" in extra:
            d = d.dropna(subset=["u0"])
        b = d.drop_duplicates("PATNO")
        for c, s in [("agec", "age0"), ("durc", "dur"), ("P0c", "P0"), ("L0c", "L0"), ("u0c", "u0")]:
            d[c] = d[s] - b[s].mean()
        m1 = fit_lmm("G ~ t*C(sg) + t*trt0 + " + COV + STEP + extra, d); m0 = fit_lmm("G ~ t*C(sg) + trt0 + " + COV + STEP + extra, d)
        return dict(n=d.PATNO.nunique(), est=m1.params["t:trt0"], p=lrt(m1, m0, 1)[1])

    OUT["matched_pairs"] = len(pairs)
    OUT["within_genetic_all"] = within_gen(set(bg.index))
    OUT["within_genetic_matched"] = within_gen(mids)
    OUT["within_genetic_matched_updrs"] = within_gen(mids, " + u0c + u0c:t")

    # treatment-state-aligned comparisons on a disease-time axis
    Z = X3.copy(); Z["tdx"] = Z.dur + Z.t; Z["on"] = (Z.LEDD > 0).astype(int)


    def state_cmp(state, gen_ids, sp_filter=None, extra="", need_u=False):
        sp = Z[(Z.subgroup == "Sporadic PD") & (Z.on == state)]
        if sp_filter is not None:
            sp = sp[sp_filter(sp)]
        d = pd.concat([sp, Z[Z.PATNO.isin(gen_ids) & (Z.on == state)]])
        d = d[d.groupby("PATNO").t.transform("nunique") >= 2].copy()
        lo = max(d[d.subgroup != "Sporadic PD"].tdx.quantile(.02), d[d.subgroup == "Sporadic PD"].tdx.quantile(.02))
        hi = min(d[d.subgroup != "Sporadic PD"].tdx.quantile(.98), d[d.subgroup == "Sporadic PD"].tdx.quantile(.98))
        d = d[(d.tdx >= lo) & (d.tdx <= hi)]; d = d[d.groupby("PATNO").t.transform("nunique") >= 2].copy()
        if need_u:
            d["u0"] = d.PATNO.map(BL.updrs3_score); d = d.dropna(subset=["u0"])
        d["gen"] = (d.subgroup != "Sporadic PD").astype(int); d["tdxc"] = d.tdx - d.tdx.mean(); d["tdx2"] = d.tdxc ** 2
        b = d.drop_duplicates("PATNO"); d["agec"] = d.age0 - b.age0.mean(); d["LEDDc"] = (d.LEDD - d.LEDD.mean()) / 100
        d["P0c"] = d.P0 - b.P0.mean()
        if need_u:
            d["u0c"] = d.u0 - b.u0.mean()
        cov = " + tdx2 + agec + C(sex)" + STEP + (" + LEDDc" if state == 1 else "") + extra
        best = {}
        for key, f in (("full", "G ~ tdxc*gen" + cov), ("null", "G ~ tdxc + gen" + cov)):
            for meth in (["powell", "lbfgs"], ["nm", "lbfgs"], ["bfgs"]):
                try:
                    r = smf.mixedlm(f, d, groups=d.PATNO, re_formula="~tdxc").fit(reml=False, method=meth)
                    if r.converged and (key not in best or r.llf > best[key].llf):
                        best[key] = r
                except Exception:
                    pass
        return dict(n_sp=int((b.gen == 0).sum()), n_gen=int((b.gen == 1).sum()),
                    diff=best["full"].params["tdxc:gen"], p=lrt(best["full"], best["null"], 1)[1])

    gt, gu = set(bg.index[bg.trt0 == 1]), set(bg.index[bg.trt0 == 0])
    OUT["on_treatment"] = state_cmp(1, gt)
    OUT["off_treatment"] = state_cmp(0, gu)
    OUT["on_treatment_phase1"] = state_cmp(1, gt, sp_filter=lambda s: s.phase == 1)
    OUT["on_treatment_putamen"] = state_cmp(1, gt, extra=" + P0c + P0c:tdxc")
    OUT["on_treatment_updrs"] = state_cmp(1, gt, extra=" + u0c + u0c:tdxc", need_u=True)
    OUT["on_treatment_variant"] = state_cmp(1, gt & var_ids)
    OUT["pairs"] = pairs
    RESULTS[MODE] = OUT
    print(MODE, "done", flush=True)
json.dump(RESULTS, open("corelab_results.json", "w"), indent=1, default=float)
print("DONE", flush=True)
