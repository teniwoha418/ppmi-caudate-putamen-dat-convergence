# Caudate–putamen DAT convergence in treated *GBA* and *LRRK2* Parkinson's disease (PPMI)

Analysis code for the manuscript:

> T. Shiiba. Faster convergence of caudate and putamen dopamine transporter binding in treated *GBA* and *LRRK2* Parkinson's disease: a PPMI study. Manuscript submitted for publication.

The code reproduces all analyses in the manuscript and its Supplementary material from the Parkinson's Progression Markers Initiative (PPMI) curated data release 20260511.

## Data access

PPMI data are **not** included in this repository. They are available to qualified researchers through the PPMI database (https://www.ppmi-info.org/access-data-specimens/download-data) under the PPMI Data Use Agreement. Two input files are required:

| File | Used by |
|---|---|
| `PPMI_Curated_Data_Cut_Public_20260511.xlsx` (sheet `20260511`) | all scripts |
| PPMI genetic consensus file (Indiana University snapshot 20251025), CSV | `design_based_comparisons.py`, `corelab_period_sensitivity.py` |

Place the input files anywhere and pass their paths on the command line. Do not commit them (they are excluded by `.gitignore`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd code
```

Tested with Python 3.11.16, NumPy 2.4.6, pandas 2.3.3, SciPy 1.17.1, statsmodels 0.15.0, lifelines 0.30.3, openpyxl 3.1.5, Matplotlib 3.11.2 and Pillow 12.3.0 (python-docx 1.2.0 is needed only for the optional Table 2 cross-check).

## Running

All scripts import `pipeline.py` from the same directory, so run them from `code/`.

```bash
XLSX=/path/to/PPMI_Curated_Data_Cut_Public_20260511.xlsx
IU=/path/to/iu_genetic_consensus_20251025.csv

python primary_treatment_cognition.py "$XLSX"
python -c "exec(open('pipeline.py').read()); import pandas as pd; A,_=build_cohort(pd.read_excel('$XLSX', sheet_name='20260511')); A.to_pickle('cohort_20260511.pkl')"
python robustness.py "$XLSX"
python design_based_comparisons.py "$XLSX" "$IU"
python attrition_ipw.py "$XLSX"
python corelab_period_sensitivity.py "$XLSX" "$IU"
python site_severity_evalues.py "$XLSX"
python period_onset_samplesize.py "$XLSX"
python genotype_specific.py "$XLSX"
python treatment_definition.py "$XLSX"
python severity_timing_and_matching.py "$XLSX"
python presubmission_checks.py "$XLSX"   # after primary_treatment_cognition.py
python make_figures.py "$XLSX"           # after primary_treatment_cognition.py, design_based_comparisons.py and severity_timing_and_matching.py
```

Each script writes its results to a JSON (or CSV) file in the working directory. Mixed models are fitted with several optimizers, and the highest-likelihood converged solution is kept, so a full run takes several minutes per script.

## Scripts and manuscript items

The scripts compute the estimates reported in the items listed; `make_figures.py` renders Figures 1 and 2 from the data and the result files.

| Script | Analyses | Manuscript items | Output |
|---|---|---|---|
| `pipeline.py` | Cohort construction, multi-optimizer mixed-model fitting, Tukey-adjusted contrasts, group slopes | Supplementary Methods M1–M3; Supplementary Fig. S1 | — (imported) |
| `primary_treatment_cognition.py` | Primary three-group and exploratory four-group models; untreated-at-enrollment subset; treatment strata; within-group treatment tests; disease-time curvature; cognitive mixed and Cox models | Fig. 1; Table 2; Supplementary Tables S1, S4, S6 | `primary_treatment_cognition_results.json` |
| `robustness.py` | Robustness suite (baseline Level/putamen adjustment, common follow-up window, time axes, phase 1, site random intercept) | Table 2; Supplementary Tables S3–S4 | `robust_3group.csv` |
| `design_based_comparisons.py` | Variant restriction, duration-matched genetic pairs, treatment-state-aligned comparisons on a disease-time axis (estimates with 95% CIs: Tukey-adjusted for the three-group contrasts, Wald for the two-group comparisons; `excess_decline` > 0 = faster decline in genetic or treated genetic PD) | Fig. 2; Table 2; Supplementary Tables S3 (section E) and S4 | `design_results.json` |
| `attrition_ipw.py` | Scan availability, attrition model, inverse-probability-weighted GEE, residual and near-floor checks | Supplementary Table S5 (parts B–C); Supplementary Methods M3, M6 | `attrition_results.json` |
| `corelab_period_sensitivity.py` | Imaging core-laboratory period: exclusion of scans from December 2024 and period step term | Supplementary Table S5 (part A); Supplementary Methods M6 | `corelab_results.json` |
| `site_severity_evalues.py` | Within-site fixed-effects model; on-treatment comparison with MDS-UPDRS III adjustment and matching; E-values; within-participant period step (PD and prodromal cohort) | Supplementary Tables S3 (section D) and S4 (section E); Supplementary Methods M4–M6 (E-values are computed but not reported) | `site_severity_evalues_results.json` |
| `period_onset_samplesize.py` | Subgroup-by-period interactions; symptom-onset time axis; E-value SD sensitivity; sample size for untreated genetic PD | Supplementary Tables S4 (section E) and S5 (part A); Discussion (sample size; E-value SD sensitivity is computed but not reported) | `period_onset_samplesize_results.json` |
| `genotype_specific.py` | Genotype-specific on-treatment comparisons and within-genotype duration-matched pairs | Supplementary Table S4 (sections C and E) | `genotype_specific_results.json` |
| `treatment_definition.py` | PDTRTMNT vs LEDD agreement; alternative treatment definition (either indicator positive) | Table 2; Supplementary Tables S1 (part C) and S4 | `treatment_definition_results.json` |
| `severity_timing_and_matching.py` | MDS-UPDRS III adjustment in the same participants; earliest vs first-scan on-treatment scores and their timing; structure and balance of the up-to-1:3 matched sample | Table 2; Supplementary Table S4 (section E); Supplementary Methods M5 | `severity_timing_and_matching_results.json` |
| `presubmission_checks.py` | Matched-pair mean durations and MDS-UPDRS III by group; earliest on-treatment MDS-UPDRS III by group; region-specific log(1+C) and log(1+P) contrasts and their share of the Gradient contrast; exact cognition LRT p | Results 3.2, 3.4–3.6; Discussion; Supplementary Table S2 (part C) | `presubmission_checks_results.json` |
| `make_figures.py` | Figure rendering: observed means and model-based trajectories (Fig. 1A; REML refit of the on-treatment disease-time model for Fig. 2A); forest plots of the estimates in `primary_treatment_cognition_results.json` (Fig. 1B), `design_results.json` and `severity_timing_and_matching_results.json` (Fig. 2B); optional cross-check against Table 2 | Figs. 1–2 | `Figure1.pdf/.tif`, `Figure2.pdf/.tif`, `figure_values.json` |

Note: the up-to-1:3 matching in `site_severity_evalues.py` and `severity_timing_and_matching.py` is greedy, and ties are broken by row order. Both scripts keep the same row order, which reproduces the reported matched sample (59 genetic and 156 sporadic participants). A different row order changes one control and the estimate by <0.0001.

## Citation

If you use this code, please cite the article (citation details will be added on publication) and this repository (see `CITATION.cff`).

## License

MIT License (see `LICENSE`). The license covers the code only, not PPMI data.

## Acknowledgement

Data used in the preparation of this work were obtained from the Parkinson's Progression Markers Initiative (PPMI) database (www.ppmi-info.org/access-data-specimens/download-data), RRID: SCR_006431. For up-to-date information on the study, visit www.ppmi-info.org.
