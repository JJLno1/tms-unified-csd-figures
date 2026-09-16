#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_all_figures.py — consolidated figure-generation script for the manuscript.
Combines and corrects all previous figure scripts into one runnable file.

Fixes vs the original cloud versions:
  1. Figure 2A uses the CORRECT coil-yaw deviation
     (delta_theta = min(theta*, 180-theta*), NOT E-field midline angle diff).
     Cut-offs: M1 30°, DLPFC 40° (verified against manuscript).
  2. All paths are configurable for local Windows execution.
  3. No cloud-only dependencies (artifact_tool replaced by pandas/openpyxl).
  4. HCP data deduplicated by subject for morphology-dependent panels.

Usage:
  python make_all_figures.py --data-dir "C:/path/to/data" --out-dir "output"
"""
import os, sys, argparse, json, time
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm

mpl.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 600,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
})
COHORT_COLORS = {'Asian': 'tab:blue', 'White': 'tab:orange', 'Black': 'tab:green'}
SERIES_COLORS = {'MSM': 'tab:blue', 'FEM original': 'tab:orange',
                 'FEM optimised': 'tab:green'}


def jitter(n, scale=0.06, seed=42):
    return np.random.default_rng(seed).normal(0, scale, n)


def save_both(fig, outdir, basename):
    fig.tight_layout()
    fig.savefig(Path(outdir) / f'{basename}.pdf', bbox_inches='tight')
    fig.savefig(Path(outdir) / f'{basename}.png', bbox_inches='tight')
    plt.close(fig)


def draw_box(ax, vals, pos, color, width=0.6, seed=42):
    bp = ax.boxplot(vals, positions=[pos], widths=width, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color='black', linewidth=1.0),
                    boxprops=dict(linewidth=1.0),
                    whiskerprops=dict(linewidth=1.0),
                    capprops=dict(linewidth=1.0))
    for p in bp['boxes']:
        p.set_facecolor('white'); p.set_edgecolor('black')
    xs = np.full(len(vals), pos) + jitter(len(vals), seed=seed)
    ax.scatter(xs, vals, s=12, alpha=0.6, color=color, edgecolors='none',
               zorder=3)


# ── Data loading ───────────────────────────────────────────────────────────
def load_data(dd):
    clin = pd.read_excel(Path(dd) / 'All new data.xlsx',
                         sheet_name='all new data')
    clin = clin[clin.subject.notna()].copy()
    clin['subject'] = clin['subject'].astype(str).str.strip()

    ang = pd.read_excel(Path(dd) / 'ALL_ANGLES_per_subject_master_table.xlsx',
                        sheet_name='per_subject_angles')
    ang['subject'] = ang['subject'].astype(str).str.strip()

    hcp = pd.read_csv(Path(dd) / 'HCP 90' / 'unified_healthy90_all.csv')
    hcp['subject'] = hcp['subject'].astype(str).str.strip()
    head = pd.read_csv(Path(dd) / '中間過程' / '顱骨和大腦體積頭圍計算'
                       / 'head_metrics_all_cohorts.csv')
    head['subject'] = head['subject'].astype(str).str.strip()
    demo = pd.read_excel(Path(dd) / 'HCP 90' / 'HCP age sex.xlsx')
    demo = demo.rename(columns={'Subject': 'subject', 'Gender': 'Sex_demo',
                                'Age_in_Yrs': 'Age_demo'})
    demo['subject'] = demo['subject'].astype(str).str.strip()

    # wide HCP (one row per subject per target)
    wide = hcp.pivot_table(index=['cohort', 'subject'], columns='target',
                           values=['scd', 'ef99p9_opt'])
    wide.columns = ['_'.join(c) for c in wide.columns]
    wide = wide.reset_index()
    cmap = {'asian': 'Asian', 'white': 'White', 'black': 'Black'}
    wide['cohort_label'] = wide['cohort'].map(cmap)
    wide = wide.merge(head.drop(columns=['cohort']), on='subject', how='left')
    wide = wide.merge(demo[['subject', 'Sex_demo', 'Age_demo']],
                      on='subject', how='left')
    wide['ratio_M1_DLPFC'] = wide['ef99p9_opt_M1'] / wide['ef99p9_opt_DLPFC']
    wide['ratio_C3_F3'] = wide['ef99p9_opt_C3'] / wide['ef99p9_opt_F3']
    wide['delta_scd_M1_DLPFC'] = wide['scd_M1'] - wide['scd_DLPFC']
    wide['delta_scd_C3_F3'] = wide['scd_C3'] - wide['scd_F3']
    return clin, ang, wide, head


# ── Figure 1: MSM vs FEM + interaction ─────────────────────────────────────
def fig1(clin, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    ax = axes[0]
    specs = [('M1', 'MSM', clin['M1_EF_Ori_MSM'], 1, 'tab:blue'),
             ('M1', 'FEM', clin['M1_EF_Ori_FEM'], 2, 'tab:orange'),
             ('DLPFC', 'MSM', clin['DLPFC_EF_Ori_MSM'], 4, 'tab:green'),
             ('DLPFC', 'FEM', clin['DLPFC_EF_Ori_FEM'], 5, 'tab:red')]
    for i, (_, _, v, pos, c) in enumerate(specs):
        draw_box(ax, v.dropna(), pos, c, seed=100 + i)
    ax.set_xlim(0.3, 5.7); ax.set_ylim(60, 340)
    ax.set_xticks([1, 2, 4, 5]); ax.set_xticklabels(['MSM', 'FEM'] * 2)
    ax.set_ylabel('Electric field (V/m)')
    ax.set_title('A. MSM vs FEM at the original orientation')
    ax.text(1.5, 296, '***', ha='center', va='center', fontsize=14)
    ax.text(4.5, 326, '***', ha='center', va='center', fontsize=14)
    ax.text(1.5, -0.075, 'M1', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=9)
    ax.text(4.5, -0.075, 'DLPFC', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=9)

    ax = axes[1]
    means, cis = {}, {}
    for tgt, cols in [('M1', ['M1_EF_Ori_MSM', 'M1_EF_Ori_FEM']),
                      ('DLPFC', ['DLPFC_EF_Ori_MSM', 'DLPFC_EF_Ori_FEM'])]:
        means[tgt] = [clin[c].mean() for c in cols]
        cis[tgt] = [stats.t.ppf(0.975, clin[c].dropna().size - 1)
                    * stats.sem(clin[c].dropna()) for c in cols]
    x = np.array([0, 1])
    ax.errorbar(x, means['M1'], yerr=cis['M1'], marker='o', capsize=4,
                color='tab:blue', label='M1')
    ax.errorbar(x, means['DLPFC'], yerr=cis['DLPFC'], marker='o', capsize=4,
                color='tab:orange', label='DLPFC')
    ax.set_xticks(x); ax.set_xticklabels(['MSM', 'FEM'])
    ax.set_ylabel('Mean EF (V/m)'); ax.set_ylim(100, 190)
    ax.set_title('B. Target × algorithm interaction')
    ax.legend(frameon=False, loc='upper left')
    ax.text(0.5, 188, 'Interaction β = −22.34 V/m; $p$ < 0.001',
            ha='center', va='top', fontsize=8)
    save_both(fig, outdir, 'Figure_1')


# ── Figure 2: coil-yaw ROC + Stokes dose ──────────────────────────────────
def minimum_yaw(theta):
    t = float(theta) % 180.0
    return min(t, 180.0 - t)


def fig2(clin, ang, outdir):
    # compute yaw deviation from the angle master table (clinical M1/DLPFC only)
    ca = ang[ang.target.isin(['M1', 'DLPFC'])
             & ang.cohort.isin(['TC38', 'new37'])].copy()
    ca['yaw_dev'] = ca.theta_star_rel_orig_deg.apply(minimum_yaw)

    # merge EF data from clinical table (correct per-target lookup)
    ef_orig_map, ef_opt_map = {}, {}
    for _, r in clin.iterrows():
        s = r['subject']
        ef_orig_map[(s, 'M1')] = r['M1_EF_Ori_FEM']
        ef_orig_map[(s, 'DLPFC')] = r['DLPFC_EF_Ori_FEM']
        ef_opt_map[(s, 'M1')] = r['M1_EF_Opt_FEM']
        ef_opt_map[(s, 'DLPFC')] = r['DLPFC_EF_Opt_FEM']
    ca['ef_orig'] = [ef_orig_map.get((r.subject, r.target), np.nan)
                     for _, r in ca.iterrows()]
    ca['ef_opt'] = [ef_opt_map.get((r.subject, r.target), np.nan)
                    for _, r in ca.iterrows()]
    ca['below95'] = (ca.ef_orig < 0.95 * ca.ef_opt).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    ax = axes[0]
    results = {}
    for tgt, color in [('M1', 'tab:blue'), ('DLPFC', 'tab:orange')]:
        g = ca[ca.target == tgt].dropna(subset=['yaw_dev', 'below95'])
        fpr, tpr, th = roc_curve(g.below95, g.yaw_dev)
        auc_v = roc_auc_score(g.below95, g.yaw_dev)
        j = tpr - fpr
        idx = int(np.argmax(j))
        cut = th[idx]
        results[tgt] = {'auc': auc_v, 'cutoff': cut, 'n': len(g)}
        ax.plot(fpr, tpr, color=color, label=f'{tgt}: AUC = {auc_v:.3f}')
        ax.scatter(fpr[idx], tpr[idx], color=color, s=45, zorder=3)
        ax.annotate(f'{tgt} cut-off\n{cut:.0f}°',
                    xy=(fpr[idx], tpr[idx]),
                    xytext=(0.23, 0.93) if tgt == 'M1' else (0.22, 0.73),
                    textcoords='data', ha='left', va='center',
                    arrowprops=dict(arrowstyle='-', lw=0.9, color='black'))
    ax.plot([0, 1], [0, 1], '--', color='tab:green', lw=1.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.set_title('A. Clinical-to-optimal coil-yaw ROC')
    ax.legend(frameon=False, loc='lower right')

    # Panel B: Stokes vs field-matched dose (unchanged from original)
    ax = axes[1]
    series = [('MSM', 'Stokes Dose (MSM)', 'Equal_Dose_Ori_MSM'),
              ('FEM original', 'Stokes Dose (FEM)', 'Equal_Dose_Ori_FEM'),
              ('FEM optimised', 'Stokes Dose (FEM)', 'Equal_Dose_Opt_FEM')]
    x_all, y_all, bias = [], [], []
    for label, xc, yc in series:
        d = clin[[xc, yc]].dropna()
        x, y = d[xc].values, d[yc].values
        x_all.extend(x); y_all.extend(y)
        ax.scatter(x, y, s=14, alpha=0.55, color=SERIES_COLORS[label],
                   label=label)
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, m * xs + b, color=SERIES_COLORS[label])
        bias.append(f'{label} {np.mean(y - x):+.2f}')
    lo = min(min(x_all), min(y_all)) - 2
    hi = max(max(x_all), max(y_all)) + 2
    ax.plot([lo, hi], [lo, hi], '--', color='tab:red', lw=1.0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi + 8)
    ax.set_xlabel('Stokes dose (% MSO)')
    ax.set_ylabel('Field-matched dose (% MSO)')
    ax.set_title('B. Stokes vs field-matched dose')
    ax.legend(frameon=False, loc='lower right')
    ax.text(0.5, 0.98,
            'Mean bias:\n' + '; '.join(bias) + ' %MSO',
            transform=ax.transAxes, ha='center', va='top', fontsize=7.5)
    save_both(fig, outdir, 'Figure_2')
    with open(Path(outdir) / 'figure2_qc.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ── Figure 3: HCP absolute EF + ratios ─────────────────────────────────────
def fig3(wide, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.9))
    order = ['Asian', 'White', 'Black']
    ax = axes[0]
    targets = ['M1', 'DLPFC', 'C3', 'F3']
    p_text = {'M1': '$p$ = 0.010', 'DLPFC': '$p$ < 0.001',
              'C3': '$p$ < 0.001', 'F3': '$p$ < 0.001'}
    centers = np.array([1, 3, 5, 7], float)
    offsets = np.array([-0.28, 0, 0.28])
    for ti, t in enumerate(targets):
        for ci, c in enumerate(order):
            vals = wide.loc[wide.cohort_label == c,
                            f'ef99p9_opt_{t}'].dropna()
            draw_box(ax, vals, centers[ti] + offsets[ci],
                     COHORT_COLORS[c], width=0.45, seed=1000 + ti * 10 + ci)
        ax.text(centers[ti], wide[f'ef99p9_opt_{t}'].max() + 18,
                p_text[t], ha='center', va='bottom', fontsize=8)
    ax.set_xlim(0.3, 7.7); ax.set_ylim(105, 300)
    ax.set_xticks(centers); ax.set_xticklabels(targets)
    ax.set_ylabel('Optimised EF99.9 (V/m)')
    ax.set_title('A. Absolute EF under uniform 75 A/µs')
    handles = [Line2D([0], [0], marker='o', linestyle='None',
                      markerfacecolor=COHORT_COLORS[c],
                      markeredgecolor='none', markersize=5, label=c)
               for c in order]
    ax.legend(handles=handles, title='Cohort', frameon=False, loc='upper right')

    ax = axes[1]
    for ri, (label, col, center, pv) in enumerate([
            ('M1/DLPFC', 'ratio_M1_DLPFC', 1.0, '$p$ = 0.211'),
            ('C3/F3', 'ratio_C3_F3', 4.0, '$p$ = 0.889')]):
        for ci, c in enumerate(order):
            vals = wide.loc[wide.cohort_label == c, col].dropna()
            draw_box(ax, vals, center + offsets[ci],
                     COHORT_COLORS[c], width=0.45, seed=2000 + ri * 10 + ci)
        ax.text(center, wide[col].max() + (0.03 if ri == 0 else 0.05),
                pv, ha='center', va='bottom', fontsize=8)
    ax.axhline(1.0, color='tab:blue', ls='--', lw=1.0)
    ax.set_xlim(0.3, 4.7); ax.set_ylim(0.65, 1.55)
    ax.set_xticks([1.0, 4.0]); ax.set_xticklabels(['M1/DLPFC', 'C3/F3'])
    ax.set_ylabel('Within-subject EF99.9 ratio')
    ax.set_title('B. Internal EF ratios')
    ax.legend(handles=handles, title='Cohort', frameon=False, loc='lower center')
    save_both(fig, outdir, 'Figure_3')


# ── Figure 4: Depth vs EF ratio ────────────────────────────────────────────
def fig4(wide, outdir):
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for label, xc, yc, color in [
            ('M1/DLPFC', 'delta_scd_M1_DLPFC', 'ratio_M1_DLPFC', 'tab:blue'),
            ('C3/F3', 'delta_scd_C3_F3', 'ratio_C3_F3', 'tab:orange')]:
        d = wide[[xc, yc]].dropna()
        x, y = d[xc].values, d[yc].values
        ax.scatter(x, y, s=16, alpha=0.45, color=color)
        model = sm.OLS(y, sm.add_constant(x)).fit()
        xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 200)
        pred = model.get_prediction(sm.add_constant(xs)).summary_frame(0.05)
        ax.plot(xs, pred['mean'].values, color=color,
                label=f'{label}: $R^2$={model.rsquared:.3f}')
        ax.fill_between(xs, pred['obs_ci_lower'], pred['obs_ci_upper'],
                        color=color, alpha=0.12)
    ax.axhline(1.0, color='tab:blue', ls='--', lw=1.0)
    ax.axvline(0.0, color='tab:blue', ls=':', lw=1.0)
    ax.set_xlabel('ΔSCD (mm)'); ax.set_ylabel('EF99.9 ratio')
    ax.legend(frameon=False, loc='upper right')
    save_both(fig, outdir, 'Figure_4')


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--out-dir', default='figures_output')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    clin, ang, wide, head = load_data(args.data_dir)
    print(f'Clinical: {len(clin)}, angles: {len(ang)}, HCP wide: {len(wide)}')

    fig1(clin, args.out_dir)
    print('Figure 1 done')
    r2 = fig2(clin, ang, args.out_dir)
    print(f'Figure 2 done: {json.dumps(r2, indent=2)}')
    fig3(wide, args.out_dir)
    print('Figure 3 done')
    fig4(wide, args.out_dir)
    print('Figure 4 done')
    print(f'All figures saved to {args.out_dir}/')


if __name__ == '__main__':
    main()
