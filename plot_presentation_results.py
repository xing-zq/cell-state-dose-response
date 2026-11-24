"""Rebuild the Chinese presentation figures from actual saved experiment results.

No simulated measurements or confidence bands. Small dots are measured wells;
lines are dose medians. With well_scores.csv present, refresh the small example
subset; otherwise use the committed subset to rebuild figures without raw data.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures/presentation'
BLUE, ORANGE, TEAL, INK, GREY = '#368CFF', '#C5663E', '#207F80', '#182F43', '#607587'
EXAMPLES = ['Ruxolitinib', 'Tofacitinib', 'Baricitinib', 'Cabozantinib']


def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=240, facecolor='white')
    plt.close(fig)


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#A8B8C7')
    ax.grid(axis='y', color='#EAF0F5', lw=.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=GREY, length=3)


def dose_axis(ax, values):
    ax.set_xscale('log')
    ax.set_xticks(values, [f'{v:.3g}' for v in values])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel('浓度（µM，对数刻度）')


def oasis_curve(ax, g, color, low, high):
    clean(ax)
    ax.axvspan(low, high, color=color, alpha=.10, zorder=0)
    ax.plot(g.dose_uM, g.ratio, 'o-', color=color, lw=2.7, ms=6)
    ax.axhline(1, color=ORANGE, ls='--', lw=1.5)
    ax.set_yscale('log')
    ax.set_ylim(.15, 30)
    ax.set_yticks([.2, .5, 1, 2, 5, 10, 20], ['0.2', '0.5', '1', '2', '5', '10', '20'])
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    dose_axis(ax, g.dose_uM)
    ax.set_ylabel('形态偏离 / 对照界线')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ['Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei'] if f in available), None)
    if family is None:
        raise RuntimeError('Install Microsoft YaHei or Noto Sans CJK SC to render Chinese labels.')
    plt.rcParams.update({'font.family': family, 'font.size': 12, 'axes.unicode_minus': False,
                         'axes.labelcolor': INK, 'text.color': INK, 'pdf.fonttype': 42})
    oa = pd.read_csv(ROOT / 'results/oasis/trajectories.csv')
    dose = pd.read_csv(ROOT / 'results/benchmark/dose_scores.csv')
    summary = json.loads((ROOT / 'results/benchmark/summary.json').read_text())
    full = ROOT / 'results/benchmark/well_scores.csv'
    subset = ROOT / 'results/benchmark/example_well_scores.csv'
    if full.exists():
        wells = pd.read_csv(full)
        wells[wells.treatment.isin(EXAMPLES)].to_csv(subset, index=False)
    wells = pd.read_csv(subset)
    # Displayed dots and dose curves must be the same data, not different runs.
    med = wells.groupby(['treatment', 'treatment_conc'])[['R', 'O']].median().reset_index()
    check = med.merge(dose, on=['treatment', 'treatment_conc'], suffixes=('_dots', '_line'), validate='one_to_one')
    assert len(check) == len(dose[dose.treatment.isin(EXAMPLES)])
    assert np.allclose(check[['R_dots', 'O_dots']], check[['R_line', 'O_line']], atol=1e-12)
    candidate_wells = wells[wells.treatment.eq('Ruxolitinib') & wells.treatment_conc.eq(.25)]
    assert len(candidate_wells) == 6 and ((candidate_wells.R >= .5) & (candidate_wells.O <= 1)).sum() == 3

    fig, ax = plt.subplots(figsize=(8.4, 4.45), layout='constrained')
    g = oa[oa.drug == 'Cyclosporin A'].sort_values('dose_uM')
    oasis_curve(ax, g, BLUE, 11, 33.5)
    ax.text(.054, 1.16, '对照界线 = 1', color=ORANGE)
    ax.annotate('11 µM：仍未越界', xy=(11, float(g.loc[g.dose_uM.eq(11), 'ratio'].iloc[0])),
                xytext=(1.0, .28), arrowprops={'arrowstyle': '-', 'color': GREY}, color=GREY)
    ax.annotate('33.5、100 µM\n连续两档越界', xy=(33.5, float(g.loc[g.dose_uM.eq(33.5), 'ratio'].iloc[0])),
                xytext=(2, 8), arrowprops={'arrowstyle': '->', 'color': ORANGE}, color=ORANGE)
    save(fig, 'cyclosporin_threshold')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.35), sharey=True, layout='constrained')
    for ax, name, cn, c, lo, hi in [(axes[0], 'Imatinib', '伊马替尼', BLUE, 3.5, 11),
                                    (axes[1], 'Amiodarone', '胺碘酮', TEAL, 11, 33.5)]:
        oasis_curve(ax, oa[oa.drug.eq(name)], c, lo, hi)
        ax.set_title(f'{cn}：{lo:g}–{hi:g} µM', color=c, weight='bold', pad=15, fontsize=16)
        ax.tick_params(axis='x', labelsize=10)
    save(fig, 'oasis_drug_comparison')

    methods = ['projection_only', 'projection_and_residual']
    labels = ['M0 只看 R', 'M1 增加 O 限制']
    controls = pd.DataFrame(summary['controls']).set_index(['method', 'condition'])
    fig, ax = plt.subplots(figsize=(11.5, 3.8), layout='constrained')
    for k, (method, label, c) in enumerate(zip(methods, labels, [BLUE, ORANGE])):
        healthy = controls.loc[(method, 'healthy')]
        storm = controls.loc[(method, 'storm-severe')]
        values = [int(healthy['n']-healthy['positive']), int(storm['positive'])]
        ypos = np.array([1, 0]) + (.18 if k == 0 else -.18)
        ax.barh(ypos, values, height=.29, color=c, label=label)
        for y, v in zip(ypos, values):
            ax.text(v+.8, y, str(v), va='center', fontsize=17, weight='bold', color=c)
    ax.set_yticks([1, 0], ['健康参照被排除', '炎症参照被误选'], fontsize=15)
    ax.set_xlim(0, 67)
    ax.set_xlabel('孔数（每种状态各 689 个留出孔；两种错误均越少越好）')
    ax.legend(ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.19), fontsize=14)
    clean(ax); ax.grid(False); ax.grid(axis='x', color='#EAF0F5')
    save(fig, 'control_tradeoff')

    rows = {r['method']: r for r in summary['fixed_split']}
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 3.7), gridspec_kw={'width_ratios': [1, 1.8]}, layout='constrained')
    for k, (method, label, c) in enumerate(zip(methods, labels, [BLUE, ORANGE])):
        r = rows[method]; y=1-k
        left.barh(y, r['selected'], color=c, height=.48)
        left.text(r['selected']+3, y, str(r['selected']), va='center', color=c, fontsize=17, weight='bold')
        for j, key in enumerate(['R_repeats', 'joint_repeats']):
            rate=r[key]/r['selected']*100; yy=1-j+(.18 if k==0 else -.18)
            right.barh(yy, rate, color=c, height=.29)
            right.text(rate+1, yy, f"{r[key]}/{r['selected']} · {rate:.1f}%", va='center', color=c, fontsize=12)
    left.set_yticks([1, 0], labels, fontsize=12);left.set_xlim(0, 165)
    left.set_title('发现组：选中了多少药物？', fontsize=15, pad=18, weight='bold')
    left.set_xlabel('药物数量');left.set_ylim(-.6,1.6)
    right.set_yticks([1, 0], ['留出 R 达标', '留出 R+O 双达标'], fontsize=12)
    right.set_xlim(0, 86);right.set_xticks([0,20,40,60,80]);right.set_ylim(-.6,1.6)
    right.set_title('锁定浓度后：多少候选能够重现？', fontsize=15, pad=18, weight='bold')
    right.set_xlabel('在各自候选中的比例（%）')
    for ax in [left,right]:
        clean(ax);ax.grid(False);ax.grid(axis='x',color='#EAF0F5')
    save(fig, 'drug_benchmark')

    def drug_curves(axes, name, individual=False):
        g=dose[dose.treatment.eq(name)].sort_values('treatment_conc')
        w=wells[wells.treatment.eq(name)]
        for ax,col,factor,cut in [(axes[0],'R',100,50),(axes[1],'O',1,1)]:
            clean(ax)
            if individual:
                # Horizontal spacing is a tiny deterministic jitter for visibility only.
                for d,ww in w.groupby('treatment_conc'):
                    ax.scatter(d*np.exp(np.linspace(-.035,.035,len(ww))),ww[col]*factor,
                               s=15,color='#A8B8C7',alpha=.75,zorder=2)
            ax.plot(g.treatment_conc,g[col]*factor,'o-',color=BLUE,lw=2.6,ms=5,zorder=3)
            ax.axhline(cut,color=ORANGE,ls='--',lw=1.4)
            passing=(g.R>=.5)&(g.O<=1)
            ax.scatter(g.loc[passing,'treatment_conc'],g.loc[passing,col]*factor,s=110,
                       facecolors='none',edgecolors=ORANGE,lw=1.8,zorder=4)
            ax.set_xscale('log');ax.set_ylabel('目标方向 R（%）' if col=='R' else '额外偏离 O')
        dose_axis(axes[1],g.treatment_conc)
        axes[0].tick_params(labelbottom=False)
        return g

    fig,axes=plt.subplots(2,1,figsize=(8.5,4.9),sharex=True,layout='constrained')
    drug_curves(axes,'Ruxolitinib',True)
    for ax in axes:ax.axvspan(.21,.30,color=BLUE,alpha=.08,zorder=0)
    save(fig,'ruxolitinib_wells')

    fig,axes=plt.subplots(2,2,figsize=(12.5,4.65),sharey='row',layout='constrained')
    for j,(name,cn) in enumerate([('Tofacitinib','托法替布'),('Baricitinib','巴瑞替尼')]):
        drug_curves(axes[:,j],name)
        axes[0,j].set_title(cn,fontsize=16,weight='bold',pad=12)
        axes[0,j].set_ylim(-5,105);axes[1,j].set_ylim(.35,1.9)
    save(fig,'jak_comparison')

    def decision_axis(ax):
        clean(ax);ax.grid(False)
        ax.fill_between([50,125],0,1,color='#EAF5F3',zorder=0)
        ax.axvline(50,color=GREY,ls='--',lw=1.2);ax.axhline(1,color=ORANGE,ls='--',lw=1.2)
        ax.set(xlim=(0,125),ylim=(0,2.1),xlabel='目标方向 R（%）',ylabel='额外偏离 O')
        ax.text(83,.12,'M1 候选区',color=TEAL,fontsize=12,ha='center')
    fig,ax=plt.subplots(figsize=(6.3,4.4),layout='constrained');decision_axis(ax)
    for name,cn,c,dy in [('Ruxolitinib','芦可替尼',BLUE,-.25),('Baricitinib','巴瑞替尼',GREY,.30),('Cabozantinib','卡博替尼',ORANGE,.20)]:
        row=dose[dose.treatment.eq(name)&dose.treatment_conc.eq(.25)].iloc[0]
        ax.scatter(row.R*100,row.O,s=80,color=c,zorder=4)
        ax.annotate(cn,(row.R*100,row.O),xytext=(row.R*100,row.O+dy),ha='center',fontsize=13,color=c)
    save(fig,'decision_examples')

    fig,ax=plt.subplots(figsize=(6.2,4.45),layout='constrained');decision_axis(ax)
    g=dose[dose.treatment.eq('Cabozantinib')].sort_values('treatment_conc')
    ax.plot(g.R*100,g.O,'o-',color=BLUE,lw=2.3)
    for row in g.itertuples():
        dx,dy={.01:(-4,-.17),.025:(-8,.17),.1:(-13,-.1),.25:(-7,.17),1.:(5,.05),2.5:(-12,.08)}[row.treatment_conc]
        ax.annotate(f'{row.treatment_conc:g}',(row.R*100,row.O),xytext=(row.R*100+dx,row.O+dy),fontsize=11,color=GREY)
    row=g[g.treatment_conc.eq(.25)].iloc[0]
    ax.scatter(row.R*100,row.O,s=150,facecolors='none',edgecolors=ORANGE,lw=2,zorder=4)
    save(fig,'cabozantinib_state_path')
    print('Saved 8 figure sets; each PNG/PDF is derived from saved numeric results.')


if __name__=='__main__':
    main()
