"""Figures generated from the numeric result files, without raw microscopy images."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
def main():
 out=ROOT/'figures';out.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
 oa=pd.read_csv(ROOT/'results/oasis/trajectories.csv')
 fig,ax=plt.subplots(figsize=(9,4.7),layout='constrained')
 for name,c in [('Cyclosporin A','#368CFF'),('Imatinib','#C5663E'),('Amiodarone','#207F80')]:
  g=oa[oa.drug==name];ax.plot(g.dose_uM,g.ratio,'o-',color=c,lw=2,label=name)
 ax.axhline(1,color='#75869B',ls='--');ax.set(xscale='log',yscale='log',xlabel='Concentration (uM)',ylabel='Morphology score / control cutoff');ax.legend(frameon=False)
 for ext in ['png','pdf']:fig.savefig(out/f'oasis_trajectories.{ext}',dpi=200)
 plt.close(fig)
 dose=pd.read_csv(ROOT/'results/benchmark/dose_scores.csv');g=dose[dose.treatment=='Ruxolitinib']
 fig,axes=plt.subplots(2,1,figsize=(8.5,4.8),sharex=True,layout='constrained')
 for ax,col,factor,cut,label in [(axes[0],'R',100,50,'Target-direction score R (%)'),(axes[1],'O',1,1,'Extra deviation O')]:
  ax.plot(g.treatment_conc,g[col]*factor,'o-',color='#368CFF',lw=2.5)
  ax.axhline(cut,color='#C5663E',ls='--');sel=(g.R>=.5)&(g.O<=1)
  ax.scatter(g.loc[sel,'treatment_conc'],g.loc[sel,col]*factor,s=110,facecolors='none',edgecolors='#C5663E',linewidths=2)
  ax.set_xscale('log');ax.set_ylabel(label)
 axes[1].set_xticks(g.treatment_conc,[f'{v:g}' for v in g.treatment_conc]);axes[1].set_xlabel('Concentration (uM)')
 for ext in ['png','pdf']:fig.savefig(out/f'ruxolitinib.{ext}',dpi=200)
 plt.close(fig)
 print('Saved OASIS and Ruxolitinib figures (PNG/PDF).')
if __name__=='__main__':main()
