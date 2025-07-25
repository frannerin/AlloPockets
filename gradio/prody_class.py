import pandas as pd
from functools import cached_property, cache

## ProDy

# In[19]:


import prody
prody.confProDy(verbosity='none')

import numpy as np


# In[20]:


class ProDyF:
    def __init__(self, cif):
        self._cif = cif

    features = [
        "prs",
        "mechstiff",
        "rmsf",
        "essa",
    ]

    @cached_property
    def _atoms(self):
        return prody.parseMMCIF(self._cif.filename)

    @property
    def _cas(self):
        return self._atoms.select("name CA")
        
    @cached_property
    def _res_df(self):
        return pd.DataFrame((
            {
                "auth_asym_id": res.getChid(),
                "auth_seq_id": str(res.getResnum()),
                "pdbx_PDB_ins_code": res.getIcode() or "?"
            }
            for res in self._atoms.iterResidues()
        ))

    def _get_df(self, colname, col, df=None):
        if df is None:
            df = pd.DataFrame(self._res_df)
        df[colname] = col
        return df

    

    @cache
    def _anm(
        self, n_modes="all", **kwargs
        # cutoff=15.0, gamma=1.0, sparse=False, kdtree=False, # buildHessian params
        # n_modes="all", zeros=False, turbo=True, nproc=0 # calcModes params; n_modes is "all" wrt default, for the downstream methods
    ):
        anm = prody.ANM()
        anm.buildHessian(self._cas, **kwargs)
        anm.calcModes(n_modes=n_modes, **kwargs)
        return anm

    
    def _prs(self, **anm_kwargs): 
        return prody.calcPerturbResponse(self._anm(**anm_kwargs), turbo=True) # turbo false doesn't work in prody 2.4.1

    def prs(self, **anm_kwargs): 
        _, effectiveness, sensitivity = self._prs(**anm_kwargs)
        df = self._get_df("prs_effectiveness", effectiveness)
        df = self._get_df("prs_sensitivity", sensitivity, df=df)
        return df


    def _mechstiff(self, **anm_kwargs):
        return prody.calcMechStiff(self._anm(**anm_kwargs), self._cas)

    def mechstiff(self, **anm_kwargs):
        meanstiff = np.mean( self._mechstiff(**anm_kwargs), axis=0 ) # from showMeanMechStiff function
        return self._get_df("mechstiff", meanstiff)


    def _rmsflucts(self, n_modes=None, **anm_kwargs):
        anm = self._anm(**anm_kwargs) 
        return prody.calcRMSFlucts(anm if n_modes is None else anm[:n_modes])

    def rmsf(self, n_modes=20, **anm_kwargs):
        return self._get_df("rmsf", self._rmsflucts(n_modes=n_modes, **anm_kwargs))

    
    def _essa(
        self, enm="gnm", **kwargs
        # lig=None, dist=4.5, lowmem=False # setSystem params
        # n_modes=10, enm='gnm', cutoff=None (10 for GNM, 15 for ANM) # scanResidues params #### probably this **kwargs will fail with extraneous kw arguments
    ):
        essa = prody.ESSA()
        essa.setSystem(self._atoms, **kwargs)
        # assert len(essa._ca) == len(self._cif.residues), "Not all CAs picked up by ESSA"
        essa.scanResidues(enm=enm, **kwargs)
        return essa

    def essa(self, **kwargs):
        essa = self._essa(**kwargs).getESSAZscores()
        return self._get_df("essa", essa)
