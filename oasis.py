"""Reproduce three OASIS concentration trajectories and the vehicle-null check."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WELL='Metadata_well_id';PLATE='Metadata_Plate';ID='Metadata_OASIS_ID';DOSE='Metadata_Concentration'
DRUGS={'OASIS011':'Cyclosporin A','OASIS937':'Imatinib','OASIS569':'Amiodarone'}
def interval(doses,above):
    starts=np.flatnonzero(above[:-1]&above[1:])
    if not len(starts):return 'not_detected'
    i=starts[0]
    return f'<= {doses[0]:.6g}' if i==0 else f'{doses[i-1]:.6g}-{doses[i]:.6g}'

def run(data_dir,out):
    out.mkdir(parents=True,exist_ok=True)
    checks={}
    for name,want in {'metadata':'6731b56f8f4fe2db31fcdf1308c305fb','cellprofiler_raw':'0cf2b9d11268c363d756e69851a1a568'}.items():
        h=hashlib.md5()
        with (data_dir/f'{name}.parquet').open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
        assert h.hexdigest()==want,(name,'data checksum differs');checks[name]=want
    meta=pd.read_parquet(data_dir/'metadata.parquet',columns=[WELL,PLATE,ID,DOSE,'Metadata_Compound']).drop_duplicates(WELL,keep=False)
    names=pq.read_schema(data_dir/'cellprofiler_raw.parquet').names
    features={}
    for channel in ['mito','dna','er']:
        compartment='nuclei' if channel=='dna' else 'cells'
        features[channel]=[n for n in names if n.lower().startswith(compartment+'_texture_') and channel in n.lower().split('_')]
    raw=pd.read_parquet(data_dir/'cellprofiler_raw.parquet',columns=[WELL]+sum(features.values(),[])).drop_duplicates(WELL,keep=False)
    frame=meta.merge(raw,on=WELL,validate='one_to_one');rows=[];summary=[]
    for drug,name in DRUGS.items():
        rng=np.random.default_rng(20261123);matched=[]
        for plate,block in frame.groupby(PLATE):
            treated=block[block[ID]==drug]
            if treated.empty:continue
            controls=block[block.Metadata_Compound=='DMSO'].sort_values(WELL)
            assert len(controls)==69 and len(treated)==1
            controls=controls.iloc[rng.permutation(len(controls))];wells=pd.concat([controls,treated]);reference=wells.iloc[:29]
            channels=[]
            for cols in features.values():
                center=reference[cols].median();scale=(reference[cols]-center).abs().median()*1.4826
                z=(wells[cols]-center)/scale.where(scale>0)
                score=z.abs().median(1).where(z.notna().sum(1)>=(scale>0).sum()/2)
                channels.append(score/score.iloc[:29].median())
            score=pd.concat(channels,axis=1).max(1).to_numpy();assert np.isfinite(score).all()
            matched.append((treated[DOSE].iloc[0],score[29:49],score[49:69],score[-1]))
        null_calls=[];local=[]
        for dose,g in pd.DataFrame(matched,columns=['dose','cal','held','drug']).groupby('dose'):
            cal=g.cal.tolist();assert 1<=len(cal)<=2
            null=cal[0] if len(cal)==1 else np.add.outer(*cal).ravel()/2
            cutoff=np.quantile(null,.95);ratio=g.drug.mean()/cutoff
            local.append({'drug':name,'id':drug,'dose_uM':dose,'n':len(g),'ratio':ratio,'above':ratio>1})
            null_calls.append(np.stack(g.held).mean(0)>cutoff)
        table=pd.DataFrame(local);d=table.dose_uM.to_numpy();a=table.above.to_numpy();nc=np.stack(null_calls)
        summary.append({'drug':name,'n_wells':int(table.n.sum()),'interval_uM':interval(d,a),
                        'blank_series_called':int((nc[:-1]&nc[1:]).any(0).sum()),'blank_series_n':20,
                        'leave_one_dose_out':{f'{d[i]:.6g}':interval(np.delete(d,i),np.delete(a,i)) for i in range(1,len(d)-1)}})
        rows.extend(local)
    pd.DataFrame(rows).to_csv(out/'trajectories.csv',index=False)
    assert summary[0]['interval_uM']=='11-33.5' and summary[0]['blank_series_called']==2
    (out/'summary.json').write_text(json.dumps({'input_md5':checks,'results':summary},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,default=Path(__file__).parent/'data/oasis');ap.add_argument('--out',type=Path,default=Path(__file__).parent/'results/oasis');a=ap.parse_args();run(a.data_dir,a.out)
