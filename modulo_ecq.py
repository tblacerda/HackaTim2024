# %% [markdown]
# ### modulo_ecq

# %% [markdown]
# #### Informações:
# Filtrar a base do Voronoi para identificar os sites com falhas de DL, que apresentam alta utilização de PRB , e coletar o IncECQ como "pontuação" para a priorização do rollout.
# 
# - ECQ_KPI < 70%
# - PERDAS_DOWNLOAD > 20%
# - IncECQ
# - TESTES_ECQ>30
# - MS:
#     - PRB UTL>40%

# %%
import pandas as pd
import numpy as np

# %% [markdown]
# #### Inputs

# %%
#Voronoi
base_voronoi=pd.read_excel('/Users/F8039139/OneDrive - TIM/HACK@TIM_2024/Inputs/data - 2024-06-04T091321.262_VORONOI.xlsx',engine='openpyxl')

# %%
from modulo_prb import prb_utl

# %% [markdown]
# #### PRB

# %%
base_voronoi=pd.merge(base_voronoi,prb_utl[['Station ID','PRB_UTL']], left_on='ENDERECO_ID', right_on='Station ID',how='left')

# %% [markdown]
# ### Filtrar

# %%
base_voronoi.loc[(base_voronoi['ECQ_KPI']<0.7) & (base_voronoi['PERDAS_DOWNLOAD']>0.2) & (base_voronoi['TESTES_ECQ']>30) & (base_voronoi['PRB_UTL']>40) , 'Filtro_ecq']='X'
base_voronoi= base_voronoi[base_voronoi.Filtro_ecq.eq('X')]

# %% [markdown]
# ### Outputs

# %%
#Escrever arquivo
writer = pd.ExcelWriter('/Users/F8039139/OneDrive - TIM/HACK@TIM_2024/Outputs/base_voronoi.xlsx', engine = 'xlsxwriter')
base_voronoi.to_excel(writer, index=False)
writer.save()
writer.close()


