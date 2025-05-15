import pandas as pd
import numpy as np

def carregar_dados():
        
    df_MS = pd.read_csv('MOC/VW_MICRO_STRATEGY.dsv', delimiter='¬',  encoding='latin-1')
    df_MS['PRB_UTIL_DL'] = (df_MS['PRB_UTIL_MEAN_DL_NUM'] / df_MS['PRB_UTIL_MEAN_DL_DEN']).round(2)
    df_MS['CAPACIDADE'] = (df_MS['PRB_UTIL_MEAN_DL_NUM'] / df_MS['PRB_UTIL_MEAN_DL_DEN']).round(2)
    df_MS = df_MS[['ENDERECO_ID', 'CAPACIDADE']]
    ####
    df_NPS = pd.read_csv('MOC/VW_NPS.dsv', delimiter='¬',  encoding='latin-1')
    df_NPS.rename(columns={'AFET_UTILIZACAO': 'NPS'}, inplace=True)
    ####
    df_NTFLW = pd.read_csv('MOC/VW_NETFLOW_ACESSO.dsv', delimiter='¬',  encoding='latin-1')
    df_NTFLW = df_NTFLW[df_NTFLW['STATUS_OC'] == 'ACTIVATED']
    df_NTFLW = df_NTFLW[df_NTFLW['REAL_ATIVACAO'].isnull()]
    df_NTFLW.rename(columns={'ORDEM_COMPLEXA': 'OC'}, inplace = True)
    #df_NTFLW.rename(columns={'BASELINE_ACORDADO': 'URGENCIA'}, inplace = True)
    df_NTFLW['URGENCIA'] = pd.to_datetime(df_NTFLW['BASELINE_ACORDADO'], format='%d/%m/%Y', errors='coerce') - pd.Timestamp('2025-01-01')
    df_NTFLW['URGENCIA'] = df_NTFLW['URGENCIA'].dt.days
    df_NTFLW.rename(columns={'FORNECEDOR': 'VENDOR'}, inplace=True)
    df_NTFLW['PLANO_VERAO'] = df_NTFLW['PLANO_VERAO'].apply(lambda x: 1 if x == 'SIM' else 0)
    df_NTFLW.rename(columns={'PLANO_VERAO': 'DEMANDA SAZONAL'}, inplace=True)
    df_NTFLW['COMERCIAL'] = df_NTFLW['PRIORIDADE'].apply(lambda x: 2 if x == 'SPECIAL PROJECTS' else (1 if x == 'ROTEIRO BLACK' else 0))
    df_NTFLW['CORPORATIVO'] = df_NTFLW['PRIORIDADE'].apply(lambda x: 1 if 'B2B' in str(x) else 0)
    df_NTFLW['OBRIGACAO'] = df_NTFLW['PRIORIDADE'].apply(lambda x: 1 if 'OBRIGACAO' in str(x) else 0)
    ####
    df_REPLAN = pd.read_csv('MOC/VW_REPLAN_ACESSO.dsv', delimiter='¬',  encoding='latin-1')
    df_REPLAN = df_REPLAN[['ID_ORDEM_COMPLEXA', 'RISCO_INFRA', 'RISCO_DETENTOR','RISCO_RFW', 'RISCO_TX']]
    df_REPLAN.rename(columns={'ID_ORDEM_COMPLEXA': 'OC'}, inplace=True)
    df_REPLAN = df_REPLAN.applymap(lambda x: 3 if x == 'Alto' else (2 if x == 'Medio' else (1 if x == 'Baixo' else x)))

    #####
    df_SPAZIO = pd.read_csv('MOC/spazio.txt', encoding='latin-1', sep='\t')
    df_SPAZIO['lat'] = df_SPAZIO['lat'].astype(float)
    df_SPAZIO['long'] = df_SPAZIO['long'].astype(float)

    #####
    df_ECQ = pd.read_excel('MOC/ecq.xlsx')
    df_ECQ
    df_ECQ = df_ECQ[['CLASSIFICACAO_GSBI', 'ENDERECO_ID', 'ECQ']]
    df_ECQ.rename(columns={'CLASSIFICACAO_GSBI': 'GSBI'}, inplace=True)
    df_ECQ['GSBI'] = df_ECQ['GSBI'].apply(lambda x: 0 if x == 'Iron' else (1 if x == 'Bronze' else (2 if x == 'Silver' else (3 if x == 'Gold' else x))))
    #####
    df_pesos = pd.read_excel('MOC/pesos.xlsx')


    df_consolidado = df_NTFLW[['OC', 'ENDERECO_ID', 'COMERCIAL', 'CORPORATIVO', 'DEMANDA SAZONAL', 'OBRIGACAO', 'URGENCIA', 'UF', 'DETENTOR', 'VENDOR']]
    df_consolidado = pd.merge(df_consolidado, df_MS, on='ENDERECO_ID', how='left')
    df_consolidado['CAPACIDADE'] = df_consolidado['CAPACIDADE'].fillna(0)
    df_consolidado = pd.merge(df_consolidado, df_SPAZIO, on='ENDERECO_ID', how='left')
    df_consolidado = pd.merge(df_consolidado, df_NPS, on='ENDERECO_ID', how='left')
    df_consolidado['NPS'] = df_consolidado['NPS'].fillna(1)
    df_consolidado = pd.merge(df_consolidado, df_ECQ, on='ENDERECO_ID', how='left') 
    df_consolidado['ECQ'] = df_consolidado['ECQ'].fillna(1)
    df_consolidado = pd.merge(df_consolidado, df_REPLAN, on='OC', how='left')
    df_consolidado['ranking'] = ' '
    df_consolidado = df_consolidado[['OC', 'COMERCIAL', 'CORPORATIVO',
                                    'DEMANDA SAZONAL', 'NPS', 'OBRIGACAO',
                                    'CAPACIDADE', 'ECQ', 'GSBI', 
                                    'RISCO_TX', 'RISCO_RFW', 'RISCO_DETENTOR', 'RISCO_INFRA',
                                    'URGENCIA','ranking', 'lat', 'long', 'VENDOR', 'UF']]
    df_consolidado['GSBI'] = df_consolidado['GSBI'].fillna(0)

    df_consolidado = pd.concat([
        pd.DataFrame({'OC': ['peso', 'tipo', 'q', 'p']}),df_consolidado], ignore_index=True)

    df_consolidado.rename(columns={'VENDOR': 'vendor'}, inplace=True)
    # Create a dictionary mapping 'criterio' to 'peso' from the second DataFrame
    peso_dict = df_pesos.set_index('criterio')['peso'].to_dict()
    tipo_dict = df_pesos.set_index('criterio')['tipo'].to_dict()
    q_dict = df_pesos.set_index('criterio')['q'].to_dict()
    p_dict = df_pesos.set_index('criterio')['p'].to_dict()


    # Update the 'peso' row (index 0) in the first DataFrame
    for col in df_consolidado.columns:
        if col in peso_dict:
            df_consolidado.loc[0, col] = peso_dict[col]
            df_consolidado.loc[1, col] = tipo_dict[col]
            df_consolidado.loc[2, col] = q_dict[col]
            df_consolidado.loc[3, col] = p_dict[col]

    df_consolidado['ranking'] = np.nan

    return df_consolidado



import pandas as pd
import numpy as np
import oracledb
d = r"C:\raiz_tiago\oracledb\instantclient_23_6"
oracledb.init_oracle_client(lib_dir=d)
import paramiko


def fetch_data_from_oracle(DIAS=60):
    DIAS = 50 

    dsn_tns = oracledb.makedsn('10.192.26.203', '1522', service_name='NTWNEGAN')
    
    connection = oracledb.connect(user='USER_NTWDEV', password='USER_NTWDEV@001', dsn=dsn_tns)
    query = """
    SELECT
        ENDERECO_ID, AVG(ECQ) as ECQ, CLASSIFICACAO_GSBI as GSBI
    FROM NTW_MABI.tb_ecq_agreg
    WHERE  DIA >= SYSDATE - 60 AND
        DIA < SYSDATE
    GROUP BY ENDERECO_ID
    """

    query_teste = """
    SELECT * FROM NTW_MABI.tb_ecq_agreg FETCH FIRST 5 ROWS ONLY
    """

    df = pd.read_sql_query(query_teste, connection)
    df.columns

    connection.close()
    return df

