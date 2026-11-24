"""Check result denominators, disjoint partitions, and locked-dose comparisons."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
B=Path(__file__).resolve().parent
def main():
 summary=json.loads((B/'results/benchmark/summary.json').read_text(encoding='utf-8'))
 assert summary['n_wells']==70384 and summary['n_drugs']==1856
 models=json.loads((B/'results/benchmark/fitted_model.json').read_text(encoding='utf-8'))
 assert len(models)==53
 for p in models:
  a,b,c=[set(p[k]) for k in ['fit_site_ids','calibration_site_ids','test_site_ids']]
  assert (len(a),len(b),len(c))==(52,26,26) and not(a&b or a&c or b&c)
 r=pd.read_csv(B/'results/benchmark/fixed_split_drug_results.csv');assert len(r)==1816*2
 for _,g in r.groupby('drug'):
  assert g.discovery_plates.nunique()==1 and g.validation_plates.nunique()==1
  assert not(set(g.iloc[0].discovery_plates.split(','))&set(g.iloc[0].validation_plates.split(',')))
 for method,g in r.groupby('method'):
  selected=g[g.selected];s=next(s for s in summary['fixed_split'] if s['method']==method)
  assert len(selected)==s['selected']
  assert int(((selected.validation_R>=.5)&(selected.validation_O<=1)).sum())==s['joint_repeats']
  assert int((selected.validation_R>=.5).sum())==s['R_repeats']
 dose=pd.read_csv(B/'results/benchmark/dose_scores.csv')
 assert len(dose)==10758
 for s in summary['full_screen']:
  flag=(dose.R>=.5)
  if s['method']=='projection_and_residual':flag&=(dose.O<=1)
  assert int(flag.sum())==s['candidate_doses'] and dose.loc[flag,'treatment'].nunique()==s['candidate_drugs']
 oa=json.loads((B/'results/oasis/summary.json').read_text(encoding='utf-8'))['results']
 assert [x['interval_uM'] for x in oa]==['11-33.5','3.5-11','11-33.5']
 print('PASS: model partitions, common splits, denominators, candidate counts and OASIS intervals.')
if __name__=='__main__':main()
