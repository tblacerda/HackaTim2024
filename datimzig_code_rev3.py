import pandas as pd
import numpy as np
import logging
import os
from typing import Tuple, Optional, Dict, Any
from pymcdm.methods import PROMETHEE_II
from pymcdm.helpers import rrankdata
from ortools.linear_solver import pywraplp
from geographiclib.geodesic import Geodesic
import tratamentoMOC

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RecommendationSystem:
    def __init__(self,
                 num_max_ranking: int = 10,
                 weight_distance: float = 0.1,
                 q_recommendation: int = 10,
                 p_recommendation: int = 30,
                 recommendation_ratio: float = 0.4):
        
        # Configurações principais
        self.num_max_ranking = num_max_ranking
        self.weight_distance = weight_distance
        self.q_recommendation = q_recommendation
        self.p_recommendation = p_recommendation
        self.recommendation_ratio = recommendation_ratio
        
        # Configurações calculadas
        self._num_recommendations = max(1, int(np.ceil(num_max_ranking * recommendation_ratio)))
        
        # Dependências
        self.geod = Geodesic.WGS84
        self.logger = logging.getLogger(self.__class__.__name__)

    def validate_dataframe(self, df: pd.DataFrame, is_recommendation: bool) -> bool:
        required_columns = ['lat', 'long', 'vendor', 'UF', 'regional'] if is_recommendation else ['ranking']
        try:
            checks = [df.iloc[i, 0] == val for i, val in enumerate(['peso', 'tipo', 'q', 'p'])]
            checks.append(set(required_columns).issubset(df.columns))
            return all(checks)
        except (IndexError, KeyError) as e:
            missing_columns = [col for col in required_columns if col not in df.columns]
            self.logger.error(f"Dataframe validation error: {e}. Missing columns: {missing_columns}")
            return False

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> Optional[float]:
        try:
            result = self.geod.Inverse(lat1, lon1, lat2, lon2)
            return round(result['s12'] / 1000, 2)
        except Exception as e:
            self.logger.error(f"Distance calculation failed: {e}")
            return None

    def run_promethee(self, data: np.ndarray, weights: np.ndarray, types: np.ndarray) -> np.ndarray:
        try:
            promethee = PROMETHEE_II('vshape_2')
            return rrankdata(promethee(data, weights, types, q=self.q_recommendation, p=self.p_recommendation))
        except Exception as e:
            self.logger.error(f"PROMETHEE calculation failed: {e}")
            raise

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        weights, types, q, p = (df.iloc[i, 1:-5].astype(float).values for i in range(4))
        data = df.iloc[4:, 1:-6].astype(float).values
        rankings = self.run_promethee(data, weights, types)
        return df.iloc[4:].assign(ranking=rankings)

class Optimizer:
    def __init__(self,
                 vendor_constraint: float = 0.1,
                 uf_constraint: float = 0.03,
                 regional_constraint: float = 0.1):
        
        self.vendor_constraint = vendor_constraint
        self.uf_constraint = uf_constraint
        self.regional_constraint = regional_constraint
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        self.logger = logging.getLogger(self.__class__.__name__)

    def setup_constraints(self,
                         variables: list,
                         df: pd.DataFrame,
                         num_max_ranking: int) -> None:
        
        constraints = {
            'vendor': self.vendor_constraint,
            'UF': self.uf_constraint,
            'regional': self.regional_constraint
        }

        for col, constraint in constraints.items():
            min_count = max(1, round(num_max_ranking * constraint))
            for category in df[col].unique():
                category_vars = [variables[i] for i, row in df.iterrows() if row[col] == category]
                if len(category_vars) >= min_count:
                    self.solver.Add(sum(category_vars) >= min_count)

    def optimize_selection(self,
                          variables: list,
                          rankings: list,
                          num_max_ranking: int) -> Dict[str, Any]:
        
        try:
            self.solver.Minimize(sum(v * r for v, r in zip(variables, rankings)))
            self.solver.Add(sum(variables) == num_max_ranking)
            result_status = self.solver.Solve()
            
            return {
                'status': result_status,
                'variables': variables,
                'solver': self.solver
            }
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            raise

def generate_recommendations(main_df: pd.DataFrame,
                            candidate_df: pd.DataFrame,
                            system: RecommendationSystem) -> pd.DataFrame:
    
    recommendations = []
    weight_ranking = 1 - system.weight_distance
    
    for _, main_row in main_df.iterrows():
        try:
            vendor_candidates = candidate_df[candidate_df['vendor'] == main_row['vendor']].copy()
            if vendor_candidates.empty:
                continue
            
            vendor_candidates['distancia'] = vendor_candidates.apply(
                lambda x: system.calculate_distance(main_row['lat'], main_row['long'], x['lat'], x['long']), axis=1)
            vendor_candidates.dropna(subset=['distancia'], inplace=True)
            
            if vendor_candidates.empty:
                continue
            
            rankings = system.run_promethee(
                vendor_candidates[['ranking', 'distancia']].values,
                np.array([weight_ranking, system.weight_distance]),
                np.array([-1, -1])
            )
            
            vendor_candidates['rankingLocal'] = rankings
            top_candidates = vendor_candidates.nsmallest(system._num_recommendations, 'rankingLocal')
            
            recommendations.append({
                'OCRecomendada': top_candidates['OC'].values[0],
                'rankingGlobal': top_candidates['ranking'].values[0],
                'distancia': top_candidates['distancia'].values[0],
                'rankingLocal': top_candidates['rankingLocal'].values[0],
                'OCPrincipal': main_row['OC']
            })
        except Exception as e:
            logging.error(f"Recommendation generation failed: {e}")
            continue
            
    return pd.DataFrame(recommendations)

def DanTIMzig_recommendation(df: pd.DataFrame,
                            system: RecommendationSystem,
                            optimizer: Optimizer) -> pd.DataFrame:
    
    try:
        if not system.validate_dataframe(df, is_recommendation=True):
            raise ValueError("Invalid dataframe structure")

        processed_df = system.process_data(df)
        processed_df = processed_df[['OC', 'ranking', 'lat', 'long', 'vendor', 'UF', 'regional']]
        ranking_completo = processed_df[['OC', 'ranking']].copy()

        variables = [optimizer.solver.IntVar(0, 1, f'x_{i}') for i in range(len(processed_df))]
        optimizer.setup_constraints(variables, processed_df, system.num_max_ranking)
        result = optimizer.optimize_selection(variables, processed_df['ranking'], system.num_max_ranking)

        selected_indices = [i for i, var in enumerate(variables) if var.solution_value() > 0]
        main_candidates = processed_df.iloc[selected_indices]
        other_candidates = processed_df.drop(selected_indices)

        recommendations = generate_recommendations(main_candidates, other_candidates, system)
        
        return format_output(df, main_candidates, recommendations, ranking_completo)
    
    except Exception as e:
        logging.error(f"Recommendation system failed: {e}")
        raise

def format_output(original_df: pd.DataFrame,
                 main_ranking: pd.DataFrame,
                 recommendations: pd.DataFrame,
                 ranking_completo: pd.DataFrame) -> pd.DataFrame:
    
    output_df = original_df.iloc[4:].copy().reset_index(drop=True)
    ranking_map = main_ranking.set_index('OC')['ranking'].to_dict()
    ranking_map_completo = ranking_completo.set_index('OC')['ranking'].to_dict()

    output_df['ranking'] = output_df['OC'].map(ranking_map)
    output_df['tipo'] = 'Fora Rank'
    output_df.loc[output_df['ranking'].notna(), 'tipo'] = 'Principal'

    if not recommendations.empty:
        oc_map = recommendations.set_index('OCRecomendada')['OCPrincipal'].to_dict()
        distancia_map = recommendations.set_index('OCRecomendada')['distancia'].to_dict()
        output_df['OCPrincipal'] = output_df['OC'].map(oc_map)
        output_df['distancia'] = output_df['OC'].map(distancia_map)
        output_df.loc[output_df['OCPrincipal'].notna(), 'tipo'] = 'Recomendada'

    output_df['ranking'] = output_df['OC'].map(ranking_map_completo)
    output_df = output_df[['OC', 'COMERCIAL', 'CORPORATIVO', 'DEMANDA SAZONAL',
                          'NPS', 'OBRIGACAO', 'CAPACIDADE', 'ECQ', 'GSBI', 
                          'RISCO_TX', 'RISCO_RFW', 'RISCO_DETENTOR', 'RISCO_INFRA',
                          'URGENCIA', 'ranking', 'lat', 'long',
                          'VENDOR', 'UF','REGIONAL', 'Tipo', 'OCPrincipal','distancia']]
    
    return output_df

if __name__ == "__main__":
    try:
        # Configuração do sistema
        recommender = RecommendationSystem(
            num_max_ranking=10,
            weight_distance=0.1,
            q_recommendation=10,
            p_recommendation=30,
            recommendation_ratio=0.4
        )
        
        # Configuração do otimizador
        optimizer = Optimizer(
            vendor_constraint=0.1,
            uf_constraint=0.03,
            regional_constraint=0.1
        )
        
        # Carregar dados
        input_df = tratamentoMOC.carregar_dados()
        
        # Processar recomendações
        output = DanTIMzig_recommendation(input_df, recommender, optimizer)
        output.to_excel('Ranking.xlsx', index=False)
        
        logging.info("Analysis completed successfully")

    except Exception as e:
        logging.error(f"Main execution failed: {e}")