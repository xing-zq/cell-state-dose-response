"""Reproduce a morphology benchmark. No efficacy/toxicity ground truth is assumed."""
from pathlib import Path
from itertools import combinations
import argparse, hashlib, json, platform
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

SEED = 20261124
FEATURES = [f'feature_{i}' for i in range(128)]
METHODS = ('projection_only', 'projection_and_residual')

def fit_score_plate(frame):
    """Fit on 26 controls/class, calibrate on 13/class; hold out 13/class."""
    rng = np.random.default_rng(SEED + int(frame.plate.iloc[0]))
    fit, calibration, test = [], [], []
    for condition in ['healthy', 'storm-severe']:
        ix = frame.index[frame.treatment.isna() & frame.disease_condition.eq(condition)]
        assert len(ix) >= 52
        ix = rng.permutation(ix)
        fit.append(ix[:26]); calibration.extend(ix[26:39]); test.extend(ix[39:52])
    hvalues, svalues = [frame.loc[ix, FEATURES].to_numpy() for ix in fit]
    h, s = hvalues.mean(0), svalues.mean(0)
    scale = np.maximum(np.sqrt((hvalues.var(0, ddof=1)+svalues.var(0, ddof=1))/2), 1e-8)
    direction = (s-h)/scale
    z = (frame[FEATURES].to_numpy()-h)/scale
    projection = z @ direction / (direction @ direction)
    residual = np.linalg.norm(z-projection[:,None]*direction, axis=1)
    scores = frame[['site_id','plate','treatment','treatment_conc','disease_condition']].copy()
    scores['R'] = 1-projection; scores['residual'] = residual
    bound = scores.loc[calibration, 'residual'].quantile(.95)
    assert bound > 0
    scores['O'] = residual/bound; scores['role'] = 'treated'
    scores.loc[frame.treatment.isna(), 'role'] = 'unused_control'
    scores.loc[np.concatenate(fit),'role'] = 'fit'
    scores.loc[calibration,'role'] = 'calibration'
    scores.loc[test,'role'] = 'test'
    assert abs(scores.loc[fit[0],'R'].mean()-1)<1e-9
    assert abs(scores.loc[fit[1],'R'].mean())<1e-9
    params={'plate':int(frame.plate.iloc[0]),'healthy_center':h.tolist(),
            'storm_center':s.tolist(),'feature_scale':scale.tolist(),'O_cutoff':float(bound),
            'fit_site_ids':frame.loc[np.concatenate(fit),'site_id'].tolist(),
            'calibration_site_ids':frame.loc[calibration,'site_id'].tolist(),
            'test_site_ids':frame.loc[test,'site_id'].tolist()}
    return scores, params

def passed(r,o,method):
    return (r>=.5) if method=='projection_only' else ((r>=.5)&(o<=1))

def bh(p):
    p=np.asarray(p,float);idx=np.argsort(p);ans=np.empty(len(p))
    ans[idx]=np.minimum(1,np.minimum.accumulate((p[idx]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    return ans

def split_benchmark(wells):
    """Both selectors use the same per-drug split and lock the selected dose."""
    controls={int(p):g.R.to_numpy() for p,g in wells[(wells.role=='test')&(wells.disease_condition=='storm-severe')].groupby('plate')}
    rank_mass=np.convolve(np.convolve(np.ones(14)/14,np.ones(14)/14),np.ones(14)/14)
    tail=np.cumsum(rank_mass[::-1])[::-1]
    choices=list(combinations(range(6),3)); rows=[]; excluded=[]
    for drug,g in wells[wells.treatment.notna()].groupby('treatment'):
        layouts=g.groupby('treatment_conc').plate.apply(lambda v:tuple(sorted(v)))
        if g.plate.nunique()!=6 or layouts.nunique()!=1:
            excluded.append(drug);continue
        plates=np.sort(g.plate.unique());doses=np.sort(g.treatment_conc.unique())
        r=g.pivot(index='treatment_conc',columns='plate',values='R').loc[doses,plates].to_numpy()
        o=g.pivot(index='treatment_conc',columns='plate',values='O').loc[doses,plates].to_numpy()
        primary=int.from_bytes(hashlib.sha256((str(SEED)+drug).encode()).digest()[:8],'little')%20
        for k,disc in enumerate(choices):
            val=[i for i in range(6) if i not in disc]
            dr,do=np.median(r[:,disc],1),np.median(o[:,disc],1)
            vr,vo=np.median(r[:,val],1),np.median(o[:,val],1)
            for method in METHODS:
                candidates=np.flatnonzero(passed(dr,do,method))
                row={'drug':drug,'method':method,'split':k,'primary':k==primary,'selected':bool(len(candidates)),
                     'dose_uM':np.nan,'validation_R':np.nan,'validation_O':np.nan,
                     'R_repeats':False,'joint_repeats':False,'own_rule_repeats':False,'p_rank':1.,
                     'discovery_plates':','.join(map(str,plates[list(disc)])),
                     'validation_plates':','.join(map(str,plates[val]))}
                if len(candidates):
                    j=candidates[0]
                    row.update(dose_uM=float(doses[j]),validation_R=float(vr[j]),validation_O=float(vo[j]),
                               R_repeats=bool(vr[j]>=.5),joint_repeats=bool((vr[j]>=.5)&(vo[j]<=1)),
                               own_rule_repeats=bool(passed(vr[j],vo[j],method)))
                    ranks=sum(1+int((controls[int(plates[k])] < r[j,k]).sum()) for k in val)
                    row['p_rank']=float(tail[ranks-3])
                rows.append(row)
    all_splits=pd.DataFrame(rows);primary=all_splits[all_splits.primary].copy()
    for method,ix in primary.groupby('method').groups.items():primary.loc[ix,'p_adjusted_BH']=bh(primary.loc[ix,'p_rank'])
    return primary,all_splits,excluded

def run(data_dir,out):
    out.mkdir(parents=True,exist_ok=True)
    manifest={'seed':SEED,'input_sha256':{},'methods':list(METHODS),
              'scope':'Retrospective internal benchmark; control-state labels only, no paired efficacy or toxicity labels.',
              'rules':{'projection_only':'median R >= 0.5','projection_and_residual':'median R >= 0.5 and median O <= 1'},
              'split':'26 fit + 13 calibration + 13 test controls per condition/plate; fixed hashed 3+3 drug split',
              'comparisons_fixed_before_this_run':True,
              'prior_data_exploration':True,
              'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}
    for name in ['metadata.csv','embeddings.csv']:
        h=hashlib.sha256()
        with (data_dir/name).open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
        manifest['input_sha256'][name]=h.hexdigest()
    expected={'metadata.csv':'8cfba1c58a738a6bd0ff0af83efbe3d323cd90a17fada114792afb365f47be8f',
              'embeddings.csv':'35672cd55acd817a57dda3fad30e19e0665be0728fb4c7387ce1a5cf90b478b6'}
    assert manifest['input_sha256']==expected,'Different data release; review protocol before running.'
    (out/'protocol.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    meta=pd.read_csv(data_dir/'metadata.csv');emb=pd.read_csv(data_dir/'embeddings.csv')
    assert meta.site_id.is_unique and emb.site_id.is_unique
    data=meta.merge(emb,on='site_id',validate='one_to_one')
    assert len(data)==len(meta)==70384 and np.isfinite(data[FEATURES].to_numpy()).all()
    scores=[];parameters=[]
    for _,g in data.groupby('plate'):
        s,p=fit_score_plate(g);scores.append(s);parameters.append(p)
    wells=pd.concat(scores).sort_index();wells.to_csv(out/'well_scores.csv',index=False)
    (out/'fitted_model.json').write_text(json.dumps(parameters),encoding='utf-8')
    del data,emb
    dose=wells[wells.treatment.notna()].groupby(['treatment','treatment_conc']).agg(R=('R','median'),O=('O','median'),n=('R','size')).reset_index()
    full=[];control=[];test=wells[wells.role=='test']
    for method in METHODS:
        flag=passed(dose.R,dose.O,method)
        full.append({'method':method,'candidate_drugs':int(dose.loc[flag,'treatment'].nunique()),'candidate_doses':int(flag.sum())})
        for condition,g in test.groupby('disease_condition'):
            positive=passed(g.R,g.O,method)
            control.append({'method':method,'condition':condition,'n':len(g),'positive':int(positive.sum()),'positive_rate':float(positive.mean())})
    pd.DataFrame(control).to_csv(out/'control_benchmark.csv',index=False)
    pd.DataFrame(full).to_csv(out/'full_screen.csv',index=False)
    dose.to_csv(out/'dose_scores.csv',index=False)
    primary,all_splits,excluded=split_benchmark(wells)
    primary.to_csv(out/'fixed_split_drug_results.csv',index=False)
    all_splits.to_csv(out/'all_splits.csv',index=False)
    summaries=[]
    for method,g in primary.groupby('method'):
        selected=g[g.selected];n=len(selected)
        summaries.append({'method':method,'eligible_drugs':len(g),'selected':n,
            'R_repeats':int(selected.R_repeats.sum()),'joint_repeats':int(selected.joint_repeats.sum()),
            'R_repeat_rate':float(selected.R_repeats.mean()),'joint_repeat_rate':float(selected.joint_repeats.mean()),
            'own_rule_repeats':int(selected.own_rule_repeats.sum()),
            'BH_R_shift_pass':int((g.p_adjusted_BH<=.05).sum()),'min_adjusted_p':float(g.p_adjusted_BH.min())})
    summary={'n_wells':len(wells),'n_plates':int(wells.plate.nunique()),'n_drugs':int(dose.treatment.nunique()),
             'control_auc_R':float(roc_auc_score(test.disease_condition.eq('healthy'),test.R)),
             'controls':control,'full_screen':full,'fixed_split':summaries,'excluded_drugs':excluded}
    sensitivity=all_splits.groupby(['method','split']).agg(selected=('selected','sum'),R_repeats=('R_repeats','sum'),joint_repeats=('joint_repeats','sum')).reset_index()
    sensitivity.to_csv(out/'split_sensitivity.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    # Regression checks against prior independent implementation of the same dual rule.
    dual=next(x for x in summaries if x['method']==METHODS[1])
    assert (dual['eligible_drugs'],dual['selected'],dual['joint_repeats'])==(1816,38,5)
    assert next(x for x in full if x['method']==METHODS[1])['candidate_drugs']==15
    print(json.dumps({k:v for k,v in summary.items() if k!='excluded_drugs'},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=Path(__file__).parent/'data/rxrx19b');parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results/benchmark')
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.data_dir,args.out)
