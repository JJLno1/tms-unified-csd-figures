# tms-unified-csd-figures

Publication-quality figure generation for the manuscript "Anatomical and
Model-Dependent Determinants of Electric-Field Dosimetry in TMS".

Companion repositories:
- [tms-unified-csd-ef999](https://github.com/JJLno1/tms-unified-csd-ef999) — FEM simulation pipeline
- [tms-unified-csd-statistics](https://github.com/JJLno1/tms-unified-csd-statistics) — statistical analyses

## Key correction

Figure 2A uses the **coil-yaw deviation** (effective turn angle,
`min(θ*, 180°−θ*)`), NOT the E-field midline-angle difference. This gives
ROC cut-offs of **30° (M1)** and **40° (DLPFC)** matching the manuscript.
The original cloud-version script used the incorrect metric (cut-offs
21.5°/16.2°).

## Usage

```bash
python make_all_figures.py \
    --data-dir "C:/path/to/data" \
    --out-dir  figures_output
```

## Input data

| File | Content |
|---|---|
| All new data.xlsx | Track 1 clinical (n=75): MSM/FEM EF, SCD, dose, demographics |
| ALL_ANGLES_per_subject_master_table.xlsx | Coil-yaw angles (θ*, handle angles) |
| unified_healthy90_all.csv | Track 2 HCP (n=90): EF99.9/SCD at 4 targets |
| head_metrics_all_cohorts.csv | HCP morphometry (eTIV, GM/WM/skull) |
| HCP age sex.xlsx | HCP demographics |

## Output figures

| Figure | Content |
|---|---|
| Figure_1 | MSM vs FEM box plots + Target × Algorithm interaction |
| Figure_2 | Coil-yaw ROC (correct effective-turn metric) + Stokes dose Bland-Altman |
| Figure_3 | HCP absolute EF99.9 + within-subject EF ratios (adjusted p) |
| Figure_4 | ΔSCD vs EF99.9 ratio regression |

All saved as PDF + PNG at 600 dpi.

## License

MIT
