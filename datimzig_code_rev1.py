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
        vendor_constraint=0.1, uf_constraint=0.03):
        # Initializing configuration parameters
        self.num_max_ranking = num_max_ranking
        self.num_recommendations = int(num_max_ranking * recommendation_ratio)        
        self.weight_distance = weight_distance
        self.q_recommendation = q_recommendation
        self.p_recommendation = p_recommendation
        self.vendor_constraint = vendor_constraint
        self.uf_constraint = uf_constraint

# Function to validate the structure of a DataFrame
def validate_dataframe(df: pd.DataFrame,
                       is_recommendation: bool) -> bool:
    # Defining required columns based on the context
    required_columns = ['lat', 'long', 'vendor', 'UF'] if is_recommendation else ['ranking']
    try:
        # Performing checks on the DataFrame structure
        checks = [df.iloc[i, 0] == val for i, val in enumerate(['peso', 'tipo', 'q', 'p'])]
        checks.append(set(required_columns).issubset(df.columns))
        return all(checks)
    except (IndexError, KeyError) as e:
        # Logging errors if validation fails
        logging.error(f"Dataframe validation error: {e}")
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
        top_candidates = vendor_candidates.nsmallest(config.num_recommendations, 'rankingLocal')
        
        # Appending the recommendations
        recommendations.append({
            'OCRecomendada': top_candidates['OC'].values[0],
            'rankingGlobal': top_candidates['ranking'].values[0],
            'distancia': top_candidates['distancia'].values[0],
            'rankingLocal': top_candidates['rankingLocal'].values[0],
            'OCPrincipal': main_row['OC']
        })
    # Returning the recommendations as a DataFrame
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
        data = df.iloc[4:, 1:-5].astype(float).values
        
        # Running the PROMETHEE method for ranking
        rankings = run_promethee(data, weights, types, q, p)
        # Processing the DataFrame with rankings
        processed_df = df.iloc[4:].assign(ranking=rankings)
        processed_df = processed_df[['OC',
                                     'ranking',
                                     'lat',
                                     'long',
                                     'vendor',
                                     'UF']].sort_values('ranking').reset_index(drop=True) 
        # Initializing the solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        # Creating variables for the solver
        variables = [solver.IntVar(0, 1, f'x_{i}') for i in range(len(processed_df))]
        
        # Setting up constraints for the solver
        setup_solver_constraints(solver,
                                 variables,
                                 processed_df,
                                 'UF',
                                 max(1, round(config.num_max_ranking * config.uf_constraint)))
        
        setup_solver_constraints(solver,
                                 variables,
                                 processed_df,
                                 'vendor',
                                 max(1, round(config.num_max_ranking * config.vendor_constraint)))
        
        # Defining the objective function for the solver
        solver.Minimize(sum(v * r for v, r in zip(variables, processed_df['ranking'])))
        solver.Add(sum(variables) == config.num_max_ranking)
        result_status = solver.Solve()
        # Solving the optimization problem
        if result_status != pywraplp.Solver.OPTIMAL:
            raise RuntimeError("No optimal solution found for constraints")
        
        # Extracting the selected indices from the solver solution
        selected_indices = [i for i, var in enumerate(variables) if var.solution_value()]
        # Splitting the DataFrame into main and other candidates
        main_candidates, other_candidates = processed_df.iloc[selected_indices], processed_df.drop(selected_indices)
        # Generating recommendations
        recommendations = generate_recommendations(main_candidates, other_candidates, config)
        # Returning the main candidates and recommendations
        return main_candidates, recommendations.nsmallest(int(config.num_recommendations), 'rankingGlobal')
    except Exception as e:
        # Logging errors if the recommendation system fails
        logging.error(f"Recommendation system failed: {e}")
        raise

# Main execution block
if __name__ == "__main__":
    try:
        # Initializing the configuration
        config = RecommendationConfig(
        num_max_ranking=200, weight_distance=0.1,
        q_recommendation=10, p_recommendation=30, recommendation_ratio=0.1,
        vendor_constraint=0.1, uf_constraint=0.1
        )
        # Reading the input DataFrame from an Excel file
        input_df = pd.read_excel('PRIORIZAR.xlsx')
        #input_df = tratamentoMOC.carregar_dados()
        input_df.columns
        # Running the recommendation system
        main_ranking, recommendations = promethee_recommendation(input_df, config)
        # Saving the results to Excel files
        main_ranking.drop_duplicates(inplace=True)
        main_ranking
        recommendations.drop_duplicates(inplace=True)
        recommendations
        
        main_ranking.to_excel('RankingPrincipal.xlsx', index=False)
        recommendations.to_excel('RankingRecomendacao.xlsx', index=False)
        # Logging success message
        logging.info("Analysis completed successfully")
        logging.info("Results saved to 'RankingPrincipal.xlsx' and 'RankingRecomendacao.xlsx'")

    except Exception as e:
        # Logging errors if the main execution fails
        logging.error(f"Main execution failed: {e}")


