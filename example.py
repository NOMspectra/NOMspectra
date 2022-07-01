#%%
import pandas as pd
import numpy as np
from mass import MassSpectrum
import copy
#%%
srfa = MassSpectrum().load('data/srfa.txt')
# %%
etalon = MassSpectrum().load('data/etalon2.txt')
# %%
e = copy.deepcopy(etalon)
s = copy.deepcopy(srfa)

cm_e = e.table['calculated_mass'].dropna().values
cm_s = s.table['calculated_mass'].dropna().values
# %%
operate = set(cm_s) - set(cm_e)
len(operate)
#%%
res = srfa.table.dropna()
for i, row in etalon.table.iterrows():
    if row['calculated_mass'] in operate:
        res = res.append(mark.append(row['calculated_mass'])
    else:
        mark.append(np.NaN)
srfa.table['calculated_mass'] = mark
srfa.table.dropna()
# %%
a = srfa & etalon
a.table
# %%# %%
srfa = srfa ^ etalon
srfa.table
# %%
srfa.table
# %%
