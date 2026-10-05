"""Primary three-group and exploratory four-group Gradient models, untreated-at-enrollment subset, treatment strata,
within-group treatment tests, sporadic disease-time curvature, and cognitive mixed and Cox models.
Run: python primary_treatment_cognition.py <PPMI_Curated_Data_Cut_Public_20260511.xlsx>
Requires pipeline.py in the same directory. Writes primary_treatment_cognition_results.json.
The robustness suite is in robustness.py (writes robust_3group.csv).
"""
import sys, json
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
exec(open("pipeline.py").read())

raw = pd.read_excel(sys.argv[1], sheet_name="20260511")
A, BL = build_cohort(raw)
assert A.PATNO.nunique() == 1149 and len(A) == 3576
OUT = {}
FG = "G ~ t*C(sg) + " + COV
FG0 = "G ~ t + C(sg) + " + COV


def primary(X, groups, df):
    full, null = fit_lmm(FG, X), fit_lmm(FG0, X)
    s, p = lrt(full, null, df)
    r = fit_lmm(FG, X, reml=True)
    return dict(chi2=s, p=p, slopes=group_slopes(r, groups).to_dict("records"),
                tukey=tukey_contrasts(r, groups).to_dict("records"))

# 1. primary three-group and four-group (PRKN) models
X3 = set_groups(A, GROUPS3); X4 = set_groups(A, GROUPS4)
OUT["primary_3group"] = primary(X3, GROUPS3, 2)
OUT["fourgroup_prkn"] = primary(X4, GROUPS4, 3)
FL, FL0 = FG.replace("G ~", "L ~"), FG0.replace("G ~", "L ~")
l1, l0 = fit_lmm(FL, X3), fit_lmm(FL0, X3); s, p = lrt(l1, l0, 2)
OUT["level_3group"] = dict(chi2=s, p=p, slopes=group_slopes(fit_lmm(FL, X3, reml=True), GROUPS3).to_dict("records"))
print("primary done", flush=True)

# 2. untreated at baseline
U = set_groups(X3[X3.trt0 == 0], GROUPS3)
OUT["untreated"] = primary(U, GROUPS3, 2)
u1, u0 = fit_lmm(FG + " + LEDDc + LEDDc:t", U), fit_lmm(FG0 + " + LEDDc + LEDDc:t", U)
OUT["untreated_ledd_p"] = lrt(u1, u0, 2)[1]

# 3. treatment strata
Q = X3.dropna(subset=["trt0"]).copy()
Q = Q[~((Q.subgroup == "Sporadic PD") & (Q.trt0 == 1))]
Q["stratum"] = np.where(Q.subgroup == "Sporadic PD", "Sporadic PD (untreated)",
                        np.where(Q.trt0 == 1, "Genetic, treated", "Genetic, untreated"))
ST = ["Sporadic PD (untreated)", "Genetic, untreated", "Genetic, treated"]
Q["sg"] = pd.Categorical(Q.stratum, categories=ST)
b = Q.drop_duplicates("PATNO")
for c, src in [("agec", "age0"), ("durc", "dur"), ("L0c", "L0")]:
    Q[c] = Q[src] - b[src].mean()
OUT["strata"] = primary(Q, ST, 2)
Q["t2"] = Q.t ** 2
OUT["strata_adj_quad_level"] = tukey_contrasts(fit_lmm(FG + " + t2 + L0c + L0c:t", Q), ST).to_dict("records")

# 4. within genetic: treatment x time
Gx = X3[X3.subgroup.isin(["GBA", "LRRK2"])].dropna(subset=["trt0"]).copy()
Gx["sg"] = pd.Categorical(Gx.subgroup, categories=["GBA", "LRRK2"])
b = Gx.drop_duplicates("PATNO"); Gx["agec"] = Gx.age0 - b.age0.mean(); Gx["durc"] = Gx.dur - b.dur.mean()
g1 = fit_lmm("G ~ t*C(sg) + t*trt0 + " + COV, Gx); g0 = fit_lmm("G ~ t*C(sg) + trt0 + " + COV, Gx)
OUT["within_genetic_trt_x_t"] = dict(est=g1.params["t:trt0"], se=g1.bse["t:trt0"], p=lrt(g1, g0, 1)[1])

# 5. within sporadic: treatment-initiation kink and disease-time curvature
S = X3[X3.subgroup == "Sporadic PD"].copy().sort_values(["PATNO", "t"])
b = S.drop_duplicates("PATNO"); S["agec"] = S.age0 - b.age0.mean(); S["durc"] = S.dur - b.dur.mean()
S["on"] = (S.LEDD > 0).astype(int)
S["yrs_on"] = (S.t - S.PATNO.map(S[S.on == 1].groupby("PATNO").t.min())).clip(lower=0).fillna(0)
S["t2"] = S.t ** 2
k1 = fit_lmm("G ~ t + on + yrs_on + " + COV, S); k2 = fit_lmm("G ~ t + t2 + on + yrs_on + " + COV, S)
OUT["sporadic_kink"] = dict(linear=(k1.params["yrs_on"], k1.pvalues["yrs_on"]), quadratic=(k2.params["yrs_on"], k2.pvalues["yrs_on"]))
S["tdx"] = S.dur + S.t; S["tdx2"] = S.tdx ** 2
msp = fit_lmm("G ~ tdx + tdx2 + agec + C(sex)", S, reml=True)
OUT["sporadic_disease_time"] = dict(tdx=msp.params["tdx"], tdx2=msp.params["tdx2"], p_tdx2=msp.pvalues["tdx2"],
                                    vertex=-msp.params["tdx"] / (2 * msp.params["tdx2"]))
print("treatment done", flush=True)

# 6. cognition
CL = raw[raw.PATNO.isin(set(X3.PATNO))].copy()
base = X3.drop_duplicates("PATNO").set_index("PATNO")
CL["t"] = CL.age_at_visit - CL.PATNO.map(base.dat_age0)
CL = CL[CL.t >= -0.5]
for c in ["subgroup", "G0", "L0", "age0", "sex", "educ", "dur", "trt0"]:
    CL[c] = CL.PATNO.map(base[c])
CL["sg"] = pd.Categorical(CL.subgroup, categories=GROUPS3)
b = CL.drop_duplicates("PATNO")
for c, src in [("agec", "age0"), ("durc", "dur"), ("educc", "educ")]:
    CL[c] = CL[src] - b[src].mean()
CL["G0z"] = (CL.G0 - b.G0.mean()) / b.G0.std(); CL["L0z"] = (CL.L0 - b.L0.mean()) / b.L0.std()
CCOV = "t*(agec + C(sex) + educc + durc + C(sg))"
cog = CL.dropna(subset=["COG_COMPOSITE_INT"]).rename(columns={"COG_COMPOSITE_INT": "cogc"})
mc = fit_lmm(f"cogc ~ t*G0z + t*L0z + {CCOV}", cog); mc0 = fit_lmm(f"cogc ~ G0z + t*L0z + {CCOV}", cog)
ci = mc.conf_int()
OUT["cognition"] = dict(n=cog.PATNO.nunique(), visits=len(cog),
                        G0z_t=(mc.params["t:G0z"], ci.loc["t:G0z", 0], ci.loc["t:G0z", 1], lrt(mc, mc0, 1)[1]),
                        L0z_t=(mc.params["t:L0z"], ci.loc["t:L0z", 0], ci.loc["t:L0z", 1], mc.pvalues["t:L0z"]),
                        G0z_baseline=(mc.params["G0z"], mc.pvalues["G0z"]))
cs = CL.dropna(subset=["cogstate"]).sort_values(["PATNO", "t"])
first = cs.groupby("PATNO").first()
normal0 = first[(first.cogstate == 1) & (first.t.abs() <= 0.5)].index
ev = cs[cs.PATNO.isin(normal0) & (cs.t > 0)]
imp = ev[ev.cogstate >= 2].groupby("PATNO").t.min(); last = ev.groupby("PATNO").t.max()
sv = pd.DataFrame({"time": imp.reindex(last.index).fillna(last), "event": last.index.isin(imp.index).astype(int)})
sv = sv.join(base[["subgroup", "G0", "L0", "age0", "sex", "educ", "dur"]]).dropna()
sv = sv[sv.time > 0]
for c in ["G0", "L0"]:
    sv[c + "z"] = (sv[c] - sv[c].mean()) / sv[c].std()
sv["male"] = (sv.sex == 1).astype(int); sv["GBA"] = (sv.subgroup == "GBA").astype(int); sv["LRRK2"] = (sv.subgroup == "LRRK2").astype(int)
cph = CoxPHFitter().fit(sv[["time", "event", "G0z", "L0z", "age0", "male", "educ", "dur", "GBA", "LRRK2"]], "time", "event")
OUT["cox"] = dict(n=len(sv), events=int(sv.event.sum()),
                  hr=cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].to_dict("index"))
print("cognition done", flush=True)

json.dump(OUT, open("primary_treatment_cognition_results.json", "w"), indent=1, default=float)
print("DONE", flush=True)
