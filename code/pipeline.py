"""Cohort construction and model helpers shared by all analysis scripts.

Source: PPMI_Curated_Data_Cut_Public_20260511.xlsx, sheet '20260511'.
Selection follows Supplementary Methods M1 of the manuscript.
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

GROUPS4 = ["Sporadic PD", "GBA", "LRRK2", "PRKN"]
GROUPS3 = ["Sporadic PD", "GBA", "LRRK2"]


def build_cohort(raw):
    """Return (scan-level DataFrame, baseline-row DataFrame indexed by PATNO)."""
    d = raw[(raw.COHORT == 1) & (raw.subgroup.isin(GROUPS4))].copy()
    blx = d[d.EVENT_ID == "BL"]
    bad = blx[((blx.subgroup == "GBA") & blx.PRIMDIAG.isin([5, 15]))
              | ((blx.subgroup == "LRRK2") & (blx.PRIMDIAG == 24))].PATNO.unique()
    d = d[~d.PATNO.isin(bad)]
    need_bl = ["age", "SEX", "duration_yrs", "age_datscan",
               "MIA_CAUDATE_BILAT", "MIA_PUTAMEN_BILAT"]
    blreq = d[d.EVENT_ID == "BL"].dropna(subset=need_bl)
    scans = (d[d.PATNO.isin(set(blreq.PATNO))]
             .dropna(subset=["age_datscan", "MIA_CAUDATE_BILAT", "MIA_PUTAMEN_BILAT"])
             .drop_duplicates(subset=["PATNO", "age_datscan"]))
    n_age = scans.groupby("PATNO").age_datscan.nunique()
    A = scans[scans.PATNO.isin(n_age[n_age >= 2].index)].copy()

    C, P = A.MIA_CAUDATE_BILAT, A.MIA_PUTAMEN_BILAT
    A["G"] = np.log((1 + C) / (1 + P))
    A["L"] = (np.log(1 + C) + np.log(1 + P)) / 2
    A["CmP"] = C - P
    first = A.sort_values("age_datscan").groupby("PATNO").first()
    A["t"] = A.age_datscan - A.PATNO.map(first.age_datscan)
    A["dat_age0"] = A.PATNO.map(first.age_datscan)

    bl = d[d.EVENT_ID == "BL"].drop_duplicates("PATNO").set_index("PATNO")
    for col, src in [("age0", "age"), ("sex", "SEX"), ("dur", "duration_yrs"),
                     ("educ", "EDUCYRS"), ("trt0", "PDTRTMNT"), ("phase", "enroll_phase"),
                     ("site", "SITE"), ("saa0", "CSFSAA")]:
        A[col] = A.PATNO.map(bl[src])
    for col, v in [("G0", "G"), ("L0", "L"), ("P0", "MIA_PUTAMEN_BILAT"),
                   ("C0", "MIA_CAUDATE_BILAT")]:
        A[col] = A.PATNO.map(first[v])
    A["agec"] = A.age0 - A.age0.mean()
    A["durc"] = A.dur - A.dur.mean()
    A["L0c"] = A.L0 - A.L0.mean()
    A["P0c"] = A.P0 - A.P0.mean()
    A["G0c"] = A.G0 - A.G0.mean()
    A["LEDDc"] = (A.LEDD.fillna(0) - A.LEDD.fillna(0).mean()) / 100.0
    A["is_bl"] = A.t == 0
    return A.reset_index(drop=True), bl


def set_groups(A, groups):
    """Subset to groups, set categorical with Sporadic PD reference, recenter."""
    X = A[A.subgroup.isin(groups)].copy()
    X["sg"] = pd.Categorical(X.subgroup, categories=groups)
    base = X.drop_duplicates("PATNO")
    for c, src in [("agec", "age0"), ("durc", "dur"), ("L0c", "L0"),
                   ("P0c", "P0"), ("G0c", "G0")]:
        X[c] = X[src] - base[src].mean()
    return X


COV = "agec + C(sex) + durc + agec:t + durc:t"


def fit_lmm(formula, data, reml=False, re="~t", groups="PATNO", vc=None):
    """Fit with several optimizers and keep the highest-likelihood converged solution."""
    best = None
    for meth in (["powell", "lbfgs"], ["nm", "lbfgs"], ["bfgs"], ["cg", "lbfgs"]):
        try:
            kw = dict(groups=data[groups], re_formula=re)
            if vc is not None:
                kw["vc_formula"] = vc
            r = smf.mixedlm(formula, data, **kw).fit(reml=reml, method=meth)
            if r.converged and (best is None or r.llf > best.llf):
                best = r
        except Exception:
            pass
    return best


def lrt(full, null, df):
    s = 2 * (full.llf - null.llf)
    return s, stats.chi2.sf(s, df)


def tukey_contrasts(res, groups, term_fmt="t:C(sg)[T.{}]"):
    """Pairwise slope contrasts (first minus second) with Tukey adjustment
    via the studentized range on the model covariance (large-df approximation)."""
    names = [term_fmt.format(g) for g in groups[1:]]
    beta = np.r_[0.0, res.params[names].values]
    V = np.zeros((len(groups), len(groups)))
    V[1:, 1:] = res.cov_params().loc[names, names].values
    out = []
    k = len(groups)
    for i in range(k):
        for j in range(i + 1, k):
            diff = beta[i] - beta[j]
            se = np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j])
            q = abs(diff) / se * np.sqrt(2)
            p = stats.studentized_range.sf(q, k, 1e6)
            half = stats.studentized_range.ppf(0.95, k, 1e6) / np.sqrt(2) * se
            out.append(dict(comparison=f"{groups[i]} - {groups[j]}", diff=diff, se=se,
                            lo=diff - half, hi=diff + half, p_tukey=p))
    return pd.DataFrame(out)


def group_slopes(res, groups, term_fmt="t:C(sg)[T.{}]"):
    """Per-group annual slope with Wald 95% CI."""
    rows = []
    V = res.cov_params()
    for g in groups:
        if g == groups[0]:
            est = res.params["t"]; var = V.loc["t", "t"]
        else:
            n = term_fmt.format(g)
            est = res.params["t"] + res.params[n]
            var = V.loc["t", "t"] + V.loc[n, n] + 2 * V.loc["t", n]
        se = np.sqrt(var)
        rows.append(dict(group=g, slope=est, lo=est - 1.96 * se, hi=est + 1.96 * se))
    return pd.DataFrame(rows)
