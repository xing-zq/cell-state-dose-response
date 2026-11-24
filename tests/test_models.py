import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark import fit_score_plate,passed,bh,FEATURES
from oasis import interval

class ModelTests(unittest.TestCase):
    def test_threshold_edges(self):
        r=np.array([.49,.5,.6]);o=np.array([.1,1.,1.01])
        np.testing.assert_array_equal(passed(r,o,'projection_only'),[False,True,True])
        np.testing.assert_array_equal(passed(r,o,'projection_and_residual'),[False,True,False])
        self.assertEqual(interval(np.array([1.,3.,10.]),np.array([False,True,True])),'1-3')
        self.assertEqual(interval(np.array([1.,3.,10.]),np.array([False,False,True])),'not_detected')
    def test_control_partitions_and_unit_invariance(self):
        rng=np.random.default_rng(10);f=pd.DataFrame(rng.normal(size=(105,128)),columns=FEATURES)
        f.loc[52:103,FEATURES]+=2
        f['plate']=1;f['site_id']=[f's{i}' for i in range(len(f))];f['treatment']=None;f['treatment_conc']=0.
        f['disease_condition']=['healthy']*52+['storm-severe']*53;f.loc[104,'treatment']='example'
        scores,p=fit_score_plate(f)
        sets=[set(p[k]) for k in ['fit_site_ids','calibration_site_ids','test_site_ids']]
        self.assertEqual([len(s) for s in sets],[52,26,26]);self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        transformed=f.copy();transformed[FEATURES]=f[FEATURES]*np.linspace(.1,10,128)+np.arange(128)
        new,_=fit_score_plate(transformed)
        np.testing.assert_allclose(scores[['R','O']],new[['R','O']],atol=1e-12)
    def test_bh(self):np.testing.assert_allclose(bh([.01,.02,.2]),[.03,.03,.2])
if __name__=='__main__':unittest.main()
