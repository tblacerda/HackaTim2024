import pandas as pd
import numpy as np
from scipy.stats.mstats import gmean

# Carregar os dados
df = pd.read_excel('AHP_consolidado.xlsx')

# Extrair lista única de critérios
criteria = pd.unique(df[['Option A', 'Option B']].values.ravel('K'))

# Inicializar dicionário para armazenar matrizes de cada usuário
users = df['usuario'].unique()
matrices = {user: pd.DataFrame(np.identity(len(criteria)), index=criteria, columns=criteria) 
            for user in users}

# Preencher as matrizes de comparação para cada usuário
for _, row in df.iterrows():
    user = row['usuario']
    a = row['Option A']
    b = row['Option B']
    ratio = row['Score A'] / row['Score B']
    
    # Preencher a comparação direta e sua recíproca
    matrices[user].loc[a, b] = ratio
    matrices[user].loc[b, a] = 1 / ratio

# Agregar matrizes usando média geométrica
aggregated_matrix = pd.DataFrame(np.identity(len(criteria)), index=criteria, columns=criteria)

for i in criteria:
    for j in criteria:
        if i != j:
            values = [matrices[user].loc[i, j] for user in users]
            aggregated_matrix.loc[i, j] = gmean(values)
            aggregated_matrix.loc[j, i] = mean = gmean([1/v for v in values])

# Calcular vetor de prioridades (autovetor principal)
eigenvalues, eigenvectors = np.linalg.eig(aggregated_matrix)
max_idx = np.argmax(eigenvalues.real)
weights = eigenvectors[:, max_idx].real
weights = np.abs(weights)  # Garantir valores positivos
weights /= weights.sum()

# Resultado formatado
resultado = pd.DataFrame({
    'Critério': criteria,
    'Peso': np.round(weights, 4)
}).sort_values('Peso', ascending=False)

print("Pesos finais dos critérios (AHP):")
print(resultado.to_string(index=False))


resultado.to_excel('pesos.xlsx')