# %% [markdown]
# ### modulo_prb

# %% [markdown]
# #### Informações:
# Calcular PRB UTL Médio por Station. PRB UTL na 1ª HMM de cada célula.

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# #### Inputs

# %%
#PRB UTL
base_prb=pd.read_excel('/Users/F8039139/OneDrive - TIM/HACK@TIM_2024/Inputs/Relatório Personalizado LTE - HMM_PRB.xlsx',engine='openpyxl')

# %% [markdown]
# ### Pivot Table

# %%
prb_utl= base_prb.pivot_table(index='Station ID', values=['PRB_UTIL_MEAN_DL_NUM','PRB_UTIL_MEAN_DL_DEN'],aggfunc='mean')
prb_utl['PRB_UTL']=prb_utl['PRB_UTIL_MEAN_DL_NUM']/prb_utl['PRB_UTIL_MEAN_DL_DEN']

# %% [markdown]
# ### Filtrar

# %%
prb_utl = prb_utl.reset_index()

# %%
prb_utl.loc[(prb_utl['PRB_UTL']>0.4), 'Filtro_prb']='X'
prb_utl= prb_utl[prb_utl.Filtro_prb.eq('X')]

# %% [markdown]
# ### Outputs

# %%
#Escrever arquivo
writer = pd.ExcelWriter('/Users/F8039139/OneDrive - TIM/HACK@TIM_2024/Inputs/bases processadas/prb_utl.xlsx', engine = 'xlsxwriter')
prb_utl.to_excel(writer, index=False)
writer.save()
writer.close()


