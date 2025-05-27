# Importing necessary libraries
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional
from pymcdm.methods import PROMETHEE_II
from pymcdm.helpers import rrankdata
from ortools.linear_solver import pywraplp
from geographiclib.geodesic import Geodesic
import tratamentoMOC
import os

current_path = os.getcwd()
print(f"Current working directory: {current_path}")
os.chdir(r'C:\Users\F8058552\OneDrive - TIM\__Automacao_de_tarefas\HACK@TIM_2024\__ENTREGAVEIS_GRUPO_XX_DANTIMZIG__\PYTHON')

# Configuring logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initializing the Geodesic object for distance calculations
g = Geodesic.WGS84

# Class to hold configuration parameters for the recommendation system
class RecommendationConfig:
    """Configuration parameters for the recommendation system"""
    def __init__(self,
        num_max_ranking=10, weight_distance=0.1,
        q_recommendation=10, p_recommendation=30, recommendation_ratio=0.4,
        vendor_constraint=0.1, uf_constraint=0.03, regional_constraint=0.1):
        # Initializing configuration parameters
        self.num_max_ranking = num_max_ranking
        self._num_recommendations = max(0, int(np.ceil(num_max_ranking * recommendation_ratio)))
        self.weight_distance = weight_distance
        self.q_recommendation = q_recommendation
        self.p_recommendation = p_recommendation
        self.vendor_constraint = vendor_constraint
        self.uf_constraint = uf_constraint
        self.regional_constraint = regional_constraint

# Function to validate the structure of a DataFrame
def validate_dataframe(df: pd.DataFrame,
                       is_recommendation: bool) -> bool:
    # Defining required columns based on the context
    required_columns = ['lat', 'long', 'vendor', 'UF', 'regional'] if is_recommendation else ['ranking']
    try:
        # Performing checks on the DataFrame structure
        checks = [df.iloc[i, 0] == val for i, val in enumerate(['peso', 'tipo', 'q', 'p'])]
        checks.append(set(required_columns).issubset(df.columns))
        return all(checks)
    except (IndexError, KeyError) as e:
        # Logging errors if validation fails and printing the problematic column
        missing_columns = [col for col in required_columns if col not in df.columns]
        logging.error(f"Dataframe validation error: {e}. Missing or problematic columns: {missing_columns}")
        return False

# Function to calculate the distance between two geographical points
def calculate_distance(lat1: float,
                       lon1: float,
                       lat2: float, 
                       lon2: float) -> Optional[float]:
    try:
        # Using the Geodesic library to calculate distance
        return round(g.Inverse(lat1, lon1, lat2, lon2)['s12'] / 1000, 2)
    except Exception as e:
        # Logging errors if distance calculation fails
        logging.error(f"Distance calculation failed: {e}")
        return None

# Function to execute the PROMETHEE II method for ranking
def run_promethee(data: np.ndarray, 
                  weights: np.ndarray,
                  types: np.ndarray,
                  q: np.ndarray, 
                  p: np.ndarray) -> np.ndarray:
    try:
        # Initializing the PROMETHEE II method
        promethee = PROMETHEE_II('vshape_2')
        # Returning the ranked data
        return rrankdata(promethee(data, weights, types, q=q, p=p))
    except Exception as e:
        # Logging errors if PROMETHEE calculation fails
        logging.error(f"PROMETHEE calculation failed: {e}")
        raise

# Function to set up constraints for the solver
def setup_solver_constraints(solver: pywraplp.Solver,
                              variables: list,
                              df: pd.DataFrame,
                              column: str, min_count: int):
    # Adding constraints based on unique categories in a column
    for category in df[column].unique():
        category_vars = [variables[i] for i, row in df.iterrows() if row[column] == category]
        if len(category_vars) >= min_count:
            solver.Add(sum(category_vars) >= min_count)

# Function to generate recommendations based on the main and candidate DataFrames
def generate_recommendations(main_df: pd.DataFrame,
                             candidate_df: pd.DataFrame,
                             config: RecommendationConfig) -> pd.DataFrame:
    # Initializing the recommendations list
    recommendations = []
    # Calculating the weight for ranking
    weight_ranking = 1 - config.weight_distance
    
    # Iterating through the main DataFrame rows
    for _, main_row in main_df.iterrows():
        try:
            # Filtering candidates based on the vendor
            vendor_candidates = candidate_df[candidate_df['vendor'] == main_row['vendor']].copy()
            if vendor_candidates.empty:
                continue
            
            # Calculating distances for the candidates
            vendor_candidates['distancia'] = vendor_candidates.apply(lambda x: calculate_distance(main_row['lat'], main_row['long'], x['lat'], x['long']), axis=1)
            vendor_candidates.dropna(subset=['distancia'], inplace=True)
            
            if vendor_candidates.empty:
                continue
            
            # Running the PROMETHEE method for ranking
            rankings = run_promethee(vendor_candidates[['ranking', 'distancia']].values,
                                    np.array([weight_ranking, config.weight_distance]),
                                    np.array([-1, -1]),
                                    np.array([0, config.q_recommendation]),
                                    np.array([0, config.p_recommendation]))
            
            # Assigning local rankings to the candidates
            vendor_candidates['rankingLocal'] = rankings
            # Selecting the top candidates based on local ranking
            top_candidates = vendor_candidates.nsmallest(config._num_recommendations, 'rankingLocal')
            
            # Appending the recommendations
            recommendations.append({
                'OCRecomendada': top_candidates['OC'].values[0],
                'rankingGlobal': top_candidates['ranking'].values[0],
                'distancia': top_candidates['distancia'].values[0],
                'rankingLocal': top_candidates['rankingLocal'].values[0],
                'OCPrincipal': main_row['OC']
            })
        # Returning the recommendations as a DataFrame
        except Exception as e:
            # Logging errors if recommendation generation fails
            logging.error(f"Recommendation generation failed: {e}")
            continue
    return pd.DataFrame(recommendations)

# Main function to execute the recommendation system
def promethee_recommendation(df: pd.DataFrame,
                             config: RecommendationConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # Validating the input DataFrame
    if not validate_dataframe(df, is_recommendation=True):
        raise ValueError("Invalid dataframe structure")
    
    try:
        # Extracting weights, types, q, and p parameters from the DataFrame
        weights, types, q, p = (df.iloc[i, 1:-5].astype(float).values for i in range(4))
        # Extracting the data for PROMETHEE
        data = df.iloc[4:, 1:-6].astype(float).values
        if data.shape[0] < config.num_max_ranking + config._num_recommendations:
            config.num_max_ranking = data.shape[0] 
            config._num_recommendations = 0
        
        if config.num_max_ranking < 1:
            raise ValueError("Number of max rankings must be at least 1.")

        # Running the PROMETHEE method for ranking
        rankings = run_promethee(data, weights, types, q, p)
        # Processing the DataFrame with rankings
        processed_df = df.iloc[4:].assign(ranking=rankings)
        processed_df = processed_df[['OC',
                                     'ranking',
                                     'lat',
                                     'long',
                                     'vendor',
                                     'UF',
                                     'regional']].sort_values('ranking').reset_index(drop=True) 
        
        ranking_completo = processed_df[['OC', 'ranking']].copy(deep = True)
        # Initializing the solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        # Creating variables for the solver
        variables = [solver.IntVar(0, 1, f'x_{i}') for i in range(len(processed_df))]
        ### automatizar 
        # Setting up constraints for the solver
        setup_solver_constraints(solver,
                                 variables,
                                 processed_df,
                                 'UF',
                                 max(0, round(config.num_max_ranking * config.uf_constraint)))
        
        setup_solver_constraints(solver,
                                 variables,
                                 processed_df,
                                 'vendor',
                                 max(0, round(config.num_max_ranking * config.vendor_constraint)))

        setup_solver_constraints(solver,
                            variables,
                            processed_df,
                            'regional',
                            max(0, round(config.num_max_ranking * config.regional_constraint)))
        
        # Defining the objective function for the solver
        solver.Minimize(sum(v * r for v, r in zip(variables, processed_df['ranking'])))
        solver.Add(sum(variables) == config.num_max_ranking)
        
        # Solving the optimization problem
        result_status = solver.Solve()
        if result_status == pywraplp.Solver.OPTIMAL:
            # If no optimal solution, get best feasible solution
            criterio = 1
        elif result_status == pywraplp.Solver.FEASIBLE:
            criterio = 0.5
            logging.warning("Optimal solution not found, using best feasible solution.")
        elif result_status == pywraplp.Solver.ABNORMAL or pywraplp.Solver.INFEASIBLE:
            # If the problem is infeasible, set a different criterion
            logging.warning("Infeasible solution found, relaxing constraints to obtain best possible solution.")
            return "Solução não encontrada. Relaxe as restrições e tente novamente.", "Sem Recomendações", ranking_completo
        else:
            criterio = 0.1
            logging.warning("No feasible solution found, relaxing constraints to obtain best possible solution.")
            selected_indices = [i for i, var in enumerate(variables) if var.solution_value() > criterio]
            main_candidates, other_candidates = processed_df.iloc[selected_indices], processed_df.drop(selected_indices)
            return main_candidates, pd.DataFrame(), ranking_completo         
        
        # Extracting the selected indices from the solver solution
        selected_indices = [i for i, var in enumerate(variables) if var.solution_value() == criterio]
        # Splitting the DataFrame into main and other candidates
        main_candidates, other_candidates = processed_df.iloc[selected_indices], processed_df.drop(selected_indices)
        # Generating recommendations
        if other_candidates.shape[0] == 0:
            logging.warning("No candidates available for recommendations.")
            return main_candidates, pd.DataFrame(), ranking_completo
        else:
            recommendations = generate_recommendations(main_candidates, other_candidates, config)
        # Returning the main candidates and recommendations
        try:
            return main_candidates, recommendations.nsmallest(int(config._num_recommendations), 'rankingGlobal'), ranking_completo
        except Exception as e:
            logging.error(f"Error while getting recommendations: {e}")
            return main_candidates, pd.DataFrame(), ranking_completo
    except Exception as e:
        # Logging errors if the recommendation system fails
        logging.error(f"Recommendation system failed: {e}")
        raise

def DanTIMzig_recommendation(df: pd.DataFrame,
                             config: RecommendationConfig) -> pd.DataFrame:
    ''' Apenas um Wrapper para a função promethee_recommendation
        a promethee_recommendation retorna os df Principal e Recomendação
        Esse metodo ajusta a saida para ser igual a entrada + as colunas 
        * 'ranking" que no input é vazia, retorna com o ranking preenchido de 1 a N,
        preenchendo completamente todas as OCs que foram enviadas.
        * 'Tipo' recebe tres valores 'Principal', 'Recomendada' ou 'Fora Rank'
        * 'OC Principal' para as OCs Tipo = Recomendada, essa coluna seria vazia tem o valor da OC
        que a recomendou
        * Distancia, para as OCs Tipo = recomendada, essa coluna tem a distancia em Km para a 
        OC Principal  
    '''

    # Preprocessing the input DataFrame
    try:
        main_ranking, recommendations, ranking_completo = promethee_recommendation(input_df, config)
        # Running the optimization model
       
        ranking_map_completo = ranking_completo.set_index('OC')['ranking'].to_dict()
        output_df = input_df.copy(deep = True)
        output_df = output_df.iloc[4:].reset_index(drop=True)
        ranking_map = main_ranking.set_index('OC')['ranking'].to_dict()

        output_df['ranking'] = output_df['OC'].map(ranking_map)
        output_df.loc[output_df['ranking'].notna(), 'tipo'] = 'Principal'
        if recommendations is not None and not recommendations.empty:
            OCRecomendada_map = recommendations.set_index('OCRecomendada')['OCPrincipal'].to_dict()
            output_df['OCPrincipal'] = output_df['OC'].map(OCRecomendada_map)
            output_df.loc[output_df['OCPrincipal'].notna(), 'tipo'] = 'Recomendada'
            distancia_map = recommendations.set_index('OCRecomendada')['distancia'].to_dict()
            output_df['distancia'] = output_df['OC'].map(distancia_map)
        else:
            output_df['OCPrincipal'] = None
            output_df['distancia'] = None

        output_df['tipo'] = output_df['tipo'].fillna('Fora Rank')
        output_df['ranking'] = output_df['OC'].map(ranking_map_completo)
        
        output_df.columns = ['OC', 'COMERCIAL', 'CORPORATIVO', 'DEMANDA SAZONAL',
                    'NPS', 'OBRIGACAO', 'CAPACIDADE', 'ECQ', 'GSBI', 
                    'RISCO_TX', 'RISCO_RFW', 'RISCO_DETENTOR', 'RISCO_INFRA',
                    'URGENCIA', 'ranking', 'lat', 'long',
                    'VENDOR', 'UF','REGIONAL', 'Tipo', 'OCPrincipal','distancia']

        return output_df
    
    except Exception as e:
        logging.error(f"Error while processing DataFrame: {e}")



# Main execution block
if __name__ == "__main__":
    try:
        # Initializing the configuration
        config = RecommendationConfig(
        num_max_ranking = 100, weight_distance = 0.1,
        q_recommendation = 10, p_recommendation = 30, recommendation_ratio = 0.1,
        vendor_constraint = 0.00, uf_constraint = 0.00, regional_constraint = 0.00
        )

  
        # Reading the input DataFrame from an Excel file
        #input_df1 = pd.read_excel('PRIORIZAR.xlsx')
        input_df2 = tratamentoMOC.carregar_dados2()
        input_df = input_df2 # input_df2
        input_df2.to_excel('input_df.xlsx', index=False)
        input_df.shape
        # Running the recommendation system
        output = DanTIMzig_recommendation(input_df, config)
        output['Tipo'].value_counts()
        output.to_excel('Ranking.xlsx', index=False)
        # Logging success message
        logging.info("Analysis completed successfully")
        logging.info("Results saved'")

    except Exception as e:
        # Logging errors if the main execution fails
        logging.error(f"Main execution failed: {e}")

