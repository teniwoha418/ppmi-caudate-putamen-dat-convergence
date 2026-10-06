"""Render Figure 1 and Figure 2 of the manuscript from the PPMI curated data release 20260511.
Run from code/ after primary_treatment_cognition.py, design_based_comparisons.py and severity_timing_and_matching.py:
    python make_figures.py <PPMI_Curated_Data_Cut_Public_20260511.xlsx> [Tables.docx for an optional Table 2 cross-check]
Writes Figure1.pdf/.tif, Figure2.pdf/.tif (600 dpi RGB, LZW) and figure_values.json.
Model estimates follow pipeline.py (Fig. 1) and design_based_comparisons.py (Fig. 2A, refitted by REML for display).
Fig. 1B slopes are read from primary_treatment_cognition_results.json; Fig. 2B estimates and 95% CIs from
design_results.json and severity_timing_and_matching_results.json.
"""
import sys, json, re
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms, matplotlib.text
from matplotlib.lines import Line2D
exec(open("pipeline.py").read())

XLSX = sys.argv[1]
TABLES = sys.argv[2] if len(sys.argv) > 2 else None
raw = pd.read_excel(XLSX, sheet_name="20260511")
A, BL = build_cohort(raw)
X3 = set_groups(A, GROUPS3)
RES = json.load(open("primary_treatment_cognition_results.json"))
OUT = {}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                     "xtick.direction": "out", "ytick.direction": "out", "mathtext.default": "regular"})
COL = {"Sporadic PD": "#4d4d4d", "GBA": "#0072b2", "LRRK2": "#d55e00", "gu": "#c28bb0", "gt": "#882255", "dark": "#1a1a1a", "grey": "#999999"}
ALPHA = 0.13
IT = {"Sporadic PD": "Sporadic PD", "GBA": r"$\it{GBA}$", "LRRK2": r"$\it{LRRK2}$"}
MM = 1 / 25.4


def fe_predict(res, rows):
    """Fixed-effect prediction and Wald 95% CI for design rows (list of dicts keyed by fe_params names)."""
    names = list(res.fe_params.index); b = res.fe_params.values
    V = res.cov_params().loc[names, names].values
    Xd = np.array([[row.get(n, 0.0) for n in names] for row in rows])
    est = Xd @ b; se = np.sqrt(np.einsum("ij,jk,ik->i", Xd, V, Xd))
    return est, est - 1.96 * se, est + 1.96 * se


def sex_weights(df, res):
    p = df.drop_duplicates("PATNO").sex.value_counts(normalize=True)
    return {n: float(p.get(float(m.group(1)) if m.group(1).replace(".", "").isdigit() else m.group(1), 0.0))
            for n in res.fe_params.index for m in [re.match(r"C\(sex\)\[T\.(.+)\]", n)] if m}


def mean_ci(v):
    v = np.asarray(v, float); m = v.mean(); h = 1.96 * v.std(ddof=1) / np.sqrt(len(v)); return m, m - h, m + h


def place_headers(fig, ax, hdrs, axleft):
    """Left-align section headers with the leftmost row label; assert the label column clears the left panel."""
    fig.canvas.draw(); rr = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    x0 = min(inv.transform(t.get_window_extent(rr))[0][0] for t in ax.get_yticklabels())
    for yy, txt in hdrs:
        tr = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
        fig.texts.append(matplotlib.text.Text(x0, yy, txt, transform=tr, ha="left", va="center", fontsize=8, fontweight="bold", color="#4d4d4d", figure=fig))
    fig.canvas.draw()
    right_A = axleft.get_position().x1
    assert x0 > right_A + 0.01, (x0, right_A)
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
    bbs = [(t.get_text(), t.get_window_extent(rr)) for t in texts]
    ov = [(a, b) for i, (a, ba) in enumerate(bbs) for b, bb in bbs[i + 1:] if ba.overlaps(bb)]
    print("label column x0=%.3f, left panel right=%.3f, text overlaps:" % (x0, right_A), ov)


def panel_letter(ax, s, fig):
    bb = ax.get_position()
    fig.text(0.008 if bb.x0 < 0.3 else bb.x0 - 0.17, 0.955, s, fontsize=10, fontweight="bold", va="top")

# ---------------- Figure 1 ----------------
r1 = fit_lmm("G ~ t*C(sg) + " + COV, X3, reml=True)
sw = sex_weights(X3, r1)
tt = np.linspace(0, 4.5, 91)
fig = plt.figure(figsize=(176.7 * MM, 75.3 * MM))
axA = fig.add_axes([0.085, 0.17, 0.30, 0.6]); axB = fig.add_axes([0.69, 0.17, 0.285, 0.6])
f1 = {"lines": {}, "points": {}}
offs = {"Sporadic PD": -0.05, "GBA": 0.0, "LRRK2": 0.05}
windows = [(0.5, 1.6), (1.6, 3.0), (3.0, 4.6)]
for g in GROUPS3:
    rows = []
    for t in tt:
        d = {"Intercept": 1.0, "t": t, **sw}
        if g != "Sporadic PD":
            d[f"C(sg)[T.{g}]"] = 1.0; d[f"t:C(sg)[T.{g}]"] = t
        rows.append(d)
    est, lo, hi = fe_predict(r1, rows)
    axA.fill_between(tt, lo, hi, color=COL[g], alpha=ALPHA, lw=0)
    axA.plot(tt, est, color=COL[g], lw=1.6)
    f1["lines"][g] = dict(t0=float(est[0]), t45=float(est[-1]))
    Xg = X3[X3.subgroup == g]
    pts = [("baseline", Xg[Xg.t == 0])] + [(f"{a}-{b}", Xg[(Xg.t >= a) & (Xg.t < b)]) for a, b in windows]
    f1["points"][g] = []
    for lab, s in pts:
        if len(s) < 30:
            continue
        m, l, h = mean_ci(s.G); x = float(s.t.mean()) + offs[g]
        axA.errorbar(x, m, yerr=[[m - l], [h - m]], fmt="o", ms=4, mfc="white", mec=COL[g], mew=1.1, ecolor=COL[g], elinewidth=1, capsize=0, zorder=3)
        f1["points"][g].append(dict(window=lab, n_scans=int(len(s)), x=x, mean=m, lo=l, hi=h))
n_by = X3.drop_duplicates("PATNO").subgroup.value_counts()
axA.legend([Line2D([], [], color=COL[g], lw=1.6) for g in GROUPS3], [f"{IT[g]} (n={n_by[g]})" for g in GROUPS3],
           frameon=False, loc="upper right", handlelength=1.6)
axA.set_xlabel("Years from baseline DAT-SPECT"); axA.set_ylabel("Gradient, log[(1+C)/(1+P)]")
axA.set_xlim(-0.3, 4.6); axA.set_ylim(-0.005, 0.19); axA.set_xticks(range(5))
# panel B: slopes
S1 = {d["group"]: d for d in RES["primary_3group"]["slopes"]}
S2 = {d["group"]: d for d in RES["strata"]["slopes"]}
nst = {"Sporadic PD (untreated)": 934, "Genetic, untreated": 61, "Genetic, treated": 128}
items = [("hdr", "All participants")] + [("pt", g, S1[g], COL[g], "o", f"{IT[g]} (n={n_by[g]})") for g in GROUPS3] + \
        [("hdr", "By treatment at enrollment")] + \
        [("pt", "su", S2["Sporadic PD (untreated)"], COL["Sporadic PD"], "s", "Sporadic PD, untreated (n=934)"),
         ("pt", "gu", S2["Genetic, untreated"], COL["gu"], "s", "Genetic, untreated (n=61)"),
         ("pt", "gt", S2["Genetic, treated"], COL["gt"], "s", "Genetic, treated (n=128)")]
y = len(items) - 1; yt, yl, yc = [], [], []; HDR1 = []
f1["slopes"] = {}
for it in items:
    if it[0] == "hdr":
        HDR1.append((y, it[1]))
    else:
        _, key, d, c, mk, lab = it
        axB.errorbar(d["slope"], y, xerr=[[d["slope"] - d["lo"]], [d["hi"] - d["slope"]]], fmt=mk, ms=5, color=c, ecolor=c, elinewidth=1.6, capsize=0)
        yt.append(y); yl.append(lab); yc.append(c); f1["slopes"][key] = d
    y -= 1
axB.axvline(0, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
axB.set_yticks(yt); axB.set_yticklabels(yl)
for tl, c in zip(axB.get_yticklabels(), yc):
    tl.set_color(c)
axB.tick_params(axis="y", length=0); axB.spines["left"].set_visible(False)
axB.set_ylim(-0.7, len(items) - 0.4); axB.set_xlim(-0.019, 0.0065)
axB.set_xticks([-0.015, -0.010, -0.005, 0.0, 0.005]); axB.set_xticklabels(["−0.015", "−0.010", "−0.005", "0.000", "0.005"])
axB.set_xlabel("Annual Gradient change (log-ratio/year)")
fig.text(0.008, 0.965, "A", fontsize=10, fontweight="bold", va="top"); fig.text(0.475, 0.965, "B", fontsize=10, fontweight="bold", va="top")
fig.text(0.035, 0.962, "Gradient declines in $\\it{GBA}$ and $\\it{LRRK2}$\nbut not in sporadic PD", fontsize=8, va="top")
fig.text(0.502, 0.962, "Gradient decline is concentrated in treated genetic PD", fontsize=8, va="top")
axA.set_yticks(np.arange(0, 0.176, 0.025)); axA.set_yticklabels([f"{v:.3f}" for v in np.arange(0, 0.176, 0.025)])
place_headers(fig, axB, HDR1, axA); fig.savefig("Figure1.pdf"); fig.savefig("Figure1.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, facecolor="white")
fig.savefig("Figure1_preview.png", dpi=200, facecolor="white")
OUT["figure1"] = f1; FIG1 = fig

# ---------------- Figure 2 ----------------
pp = X3.drop_duplicates("PATNO").set_index("PATNO")
bg = pp[pp.subgroup != "Sporadic PD"].dropna(subset=["trt0"])
gt = set(bg.index[bg.trt0 == 1])
Z = X3.copy(); Z["tdx"] = Z.dur + Z.t; Z["on"] = (Z.LEDD > 0).astype(int)
sp = Z[(Z.subgroup == "Sporadic PD") & (Z.on == 1)]
d = pd.concat([sp, Z[Z.PATNO.isin(gt) & (Z.on == 1)]])
d = d[d.groupby("PATNO").t.transform("nunique") >= 2].copy()
lo_ = max(d[d.subgroup != "Sporadic PD"].tdx.quantile(.02), d[d.subgroup == "Sporadic PD"].tdx.quantile(.02))
hi_ = min(d[d.subgroup != "Sporadic PD"].tdx.quantile(.98), d[d.subgroup == "Sporadic PD"].tdx.quantile(.98))
d = d[(d.tdx >= lo_) & (d.tdx <= hi_)]; d = d[d.groupby("PATNO").t.transform("nunique") >= 2].copy()
d["gen"] = (d.subgroup != "Sporadic PD").astype(int); mt = d.tdx.mean(); d["tdxc"] = d.tdx - mt; d["tdx2"] = d.tdxc ** 2
b = d.drop_duplicates("PATNO"); d["agec"] = d.age0 - b.age0.mean(); d["LEDDc"] = (d.LEDD - d.LEDD.mean()) / 100
best = None
for meth in (["powell", "lbfgs"], ["nm", "lbfgs"], ["bfgs"]):
    try:
        r = smf.mixedlm("G ~ tdxc*gen + tdx2 + agec + C(sex) + LEDDc", d, groups=d.PATNO, re_formula="~tdxc").fit(reml=True, method=meth)
        if r.converged and (best is None or r.llf > best.llf):
            best = r
    except Exception:
        pass
r2 = best; sw2 = sex_weights(d, r2)
f2 = dict(range=[float(lo_), float(hi_)], n_sporadic=int((b.gen == 0).sum()), n_genetic=int((b.gen == 1).sum()), mean_tdx=float(mt),
          slope_at_mean=dict(sporadic=float(r2.fe_params["tdxc"]), genetic=float(r2.fe_params["tdxc"] + r2.fe_params["tdxc:gen"])), lines={}, points={})
grid = np.linspace(lo_, hi_, 101)
fig2 = plt.figure(figsize=(175.6 * MM, 82.6 * MM))
ax2A = fig2.add_axes([0.085, 0.16, 0.30, 0.62]); ax2B = fig2.add_axes([0.69, 0.16, 0.285, 0.64])
spec = {0: ("Sporadic PD, on treatment", COL["Sporadic PD"], "o", "white"), 1: ("Genetic PD, treated at enrollment", COL["gt"], "s", COL["gt"])}
edges = [lo_] + [e for e in range(int(np.floor(lo_)) + 1, int(np.ceil(hi_)))] + [hi_ + 1e-9]
handles = []
for gv in (0, 1):
    lab, c, mk, fc = spec[gv]
    rows = [{"Intercept": 1.0, "tdxc": x - mt, "gen": gv, "tdxc:gen": (x - mt) * gv, "tdx2": (x - mt) ** 2, **sw2} for x in grid]
    est, lo, hi = fe_predict(r2, rows)
    ax2A.fill_between(grid, lo, hi, color=c, alpha=ALPHA + 0.01, lw=0); ax2A.plot(grid, est, color=c, lw=1.6)
    f2["lines"][lab] = dict(start=float(est[0]), end=float(est[-1]))
    dg = d[d.gen == gv]; f2["points"][lab] = []
    for a, bnd in zip(edges[:-1], edges[1:]):
        s = dg[(dg.tdx >= a) & (dg.tdx < bnd)]
        if len(s) < 20:
            continue
        m, l, h = mean_ci(s.G); x = float(s.tdx.mean())
        ax2A.errorbar(x, m, yerr=[[m - l], [h - m]], fmt=mk, ms=4, mfc=fc, mec=c, mew=1.1, ecolor=c, elinewidth=1, capsize=0, zorder=3)
        f2["points"][lab].append(dict(bin=[float(a), float(bnd)], n_scans=int(len(s)), x=x, mean=m, lo=l, hi=h))
    handles.append(Line2D([], [], color=c, marker=mk, mfc=fc, mec=c, lw=0, ms=4))
ax2A.legend(handles, [spec[0][0], spec[1][0]], frameon=False, loc="upper left", bbox_to_anchor=(0.03, 1.0), handletextpad=0.4)
ax2A.set_xlabel("Years since diagnosis (on treatment)"); ax2A.set_ylabel("Gradient, log[(1+C)/(1+P)]")
ax2A.set_xlim(0.9, 6.4); ax2A.set_ylim(-0.012, 0.205); ax2A.set_xticks(range(1, 7))
ax2A.set_yticks(np.arange(0, 0.201, 0.025)); ax2A.set_yticklabels([f"{v:.3f}" for v in np.arange(0, 0.201, 0.025)])
# panel B: estimates and 95% CIs written by design_based_comparisons.py and severity_timing_and_matching.py
def json_values():
    D = json.load(open("design_results.json")); S = json.load(open("severity_timing_and_matching_results.json"))
    g = lambda r: [r.get("excess_decline", r.get("diff")), r["lo"], r["hi"]]
    return dict(on=g(D["on_treatment"]), phase1=g(D["on_treatment_phase1"]), putamen=g(D["on_treatment_putamen"]),
                updrs_on=g(D["on_treatment_updrs"]), variant=g(D["on_treatment_variant"]), ledd0=g(D["off_treatment"]),
                all_gen=g(D["within_genetic_all"]), matched=g(D["within_genetic_matched"]),
                updrs_unadj=g(S["matched_updrs_subset_unadjusted"]), updrs_adj=g(D["within_genetic_matched_updrs"]))


def t2_values(path):
    """Optional cross-check against Table 2 of the manuscript (Tables.docx)."""
    import docx
    t = docx.Document(path).tables[1]
    rows = [[c.text.strip() for c in r.cells] for r in t.rows]
    def find(analysis_start, comparison_start):
        hit = [r for r in rows if r[1].strip().startswith(analysis_start) and r[3].startswith(comparison_start)]
        assert len(hit) == 1, (analysis_start, len(hit))
        m = re.match(r"(−?[\d.]+) \((−?[\d.]+) to (−?[\d.]+)\)", hit[0][4])
        return [float(v.replace("−", "-")) for v in m.groups()]
    return dict(
        on=find("Same treatment state, years-since-diagnosis axis", "Genetic (treated)"),
        phase1=find("sporadic PD limited to phase 1", "Genetic (treated)"),
        putamen=find("+ baseline putamen SBR × disease time", "Genetic (treated)"),
        updrs_on=find("+ baseline MDS-UPDRS III × disease time", "Genetic (treated)"),
        variant=find("GBA N409S / LRRK2 G2019S heterozygotes", "Genetic (treated)"),
        ledd0=find("Same treatment state, scans with LEDD = 0", "Genetic vs sporadic"),
        all_gen=find("All genetic participants, within-genetic model", "Treated vs untreated"),
        matched=find("1:1 duration-matched pairs (genetic PD)", "Treated vs untreated"),
        updrs_unadj=find("Matched pairs with baseline MDS-UPDRS III, unadjusted", "Treated vs untreated"),
        updrs_adj=find("Same participants + baseline MDS-UPDRS III × time", "Treated vs untreated"))


T2 = json_values()
if TABLES:
    TT = t2_values(TABLES)
    bad = {k: (np.round(T2[k], 5).tolist(), TT[k]) for k in T2 if max(abs(a - b) for a, b in zip(T2[k], TT[k])) > 6e-6}
    print("Table 2 cross-check mismatches:", bad)
items2 = [("hdr", "Genetic vs sporadic, same treatment"),
          ("pt", "on", "On treatment (75 vs 516)"), ("pt", "phase1", "  Sporadic PD phase 1 only"),
          ("pt", "putamen", "  + baseline putamen SBR"), ("pt", "updrs_on", "  + baseline MDS-UPDRS III"),
          ("pt", "variant", r"  $\it{GBA}$ N409S / $\it{LRRK2}$ G2019S only"), ("pt", "ledd0", "Scans with LEDD = 0 (13 vs 349)"),
          ("hdr", "Genetic: treated vs untreated"),
          ("pt", "all_gen", "All (128 vs 61)"), ("pt", "matched", "Duration-matched pairs (46 vs 46)"),
          ("pt", "updrs_unadj", "MDS-UPDRS III subset (n=81)"), ("pt", "updrs_adj", "  + baseline MDS-UPDRS III")]
y = len(items2) - 1; yt, yl, yc = [], [], []; HDR2 = []
for it in items2:
    if it[0] == "hdr":
        HDR2.append((y, it[1]))
    else:
        e, l, h = T2[it[1]]; c = COL["grey"] if it[1] == "ledd0" else COL["dark"]
        ax2B.errorbar(e, y, xerr=[[e - l], [h - e]], fmt="o", ms=5, color=c, ecolor=c, elinewidth=1.6, capsize=0)
        yt.append(y); yl.append(it[2]); yc.append(c)
    y -= 1
ax2B.axvline(0, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
ax2B.set_yticks(yt); ax2B.set_yticklabels(yl)
for tl, c in zip(ax2B.get_yticklabels(), yc):
    tl.set_color("#4d4d4d" if c == COL["dark"] else c)
ax2B.tick_params(axis="y", length=0); ax2B.spines["left"].set_visible(False)
ax2B.set_ylim(-0.7, len(items2) - 0.3); ax2B.set_xlim(-0.0145, 0.0245)
ax2B.set_xticks([-0.01, 0, 0.01, 0.02]); ax2B.set_xticklabels(["−0.01", "0", "0.01", "0.02"])
ax2B.set_xlabel("Excess annual Gradient decline\n(log-ratio/year)")
ax2B.text(0.99, len(items2) - 0.75, "faster decline →", transform=ax2B.get_yaxis_transform(), ha="right", va="center", fontsize=7, color="#666666")
fig2.text(0.008, 0.965, "A", fontsize=10, fontweight="bold", va="top"); fig2.text(0.475, 0.965, "B", fontsize=10, fontweight="bold", va="top")
fig2.text(0.035, 0.962, "On treatment, Gradient declines faster in genetic PD\nover a shared disease-duration range", fontsize=8, va="top")
fig2.text(0.502, 0.962, "On-treatment difference persists after adjustment;\nwithin-genetic contrast attenuated by motor severity", fontsize=8, va="top")
place_headers(fig2, ax2B, HDR2, ax2A); fig2.savefig("Figure2.pdf"); fig2.savefig("Figure2.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"}, facecolor="white")
fig2.savefig("Figure2_preview.png", dpi=200, facecolor="white")

# matplotlib writes RGBA TIFFs; flatten to RGB on white and keep LZW / 600 dpi
from PIL import Image
for _f in ("Figure1.tif", "Figure2.tif"):
    _im = Image.open(_f); _bg = Image.new("RGB", _im.size, "white"); _bg.paste(_im, mask=_im.split()[3])
    _bg.save(_f, compression="tiff_lzw", dpi=(600, 600))
OUT["figure2"] = f2; OUT["figure2B"] = T2
json.dump(OUT, open("figure_values.json", "w"), indent=1, default=float)
print("figures done", flush=True)
