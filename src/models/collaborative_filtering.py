import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

class BaselineRecommender:
    """
    Prediz sempre a média global dos ratings.
    Métodos:
        * fit(ratings_df: pd.DataFrame) -> BaselineRecommender: Treina o modelo.
        * predict(user_id: int, item_id: int) -> float: Prediz a avaliação para um par usuário-item.
        * evaluate(test_df: pd.DataFrame) -> dict: Avalia o modelo usando RMSE e MAE.
    """
    
    def __init__(self):
        self.global_mean = None
    
    def fit(self, ratings_df: pd.DataFrame) -> 'BaselineRecommender':
        """
        Treina o modelo calculando a média global dos ratings.
        Parâmetros:
            * ratings_df: pd.DataFrame -> DataFrame contendo as avaliações com colunas ['user_id', 'item_id', 'rating'].
        Retorna:
            * self: BaselineRecommender -> Instância do modelo treinado.
        """

        self.global_mean = ratings_df['rating'].mean()
        print(f"   Média global treinada: {self.global_mean:.2f}")
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Prediz a avaliação para um par usuário-item.
        Parâmetros:
            * user_id: int -> ID do usuário.
            * item_id: int -> ID do item.
        Retorna:
            * float -> Avaliação predita (média global).
        """

        return self.global_mean
    
    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """
        Avalia o modelo usando RMSE e MAE.
        Parâmetros:
            * test_df: pd.DataFrame -> DataFrame de teste com colunas ['user_id', 'item_id', 'rating'].
        Retorna:
            * dict -> Dicionário com métricas {'RMSE': float, 'MAE': float}.
        """
        
        predictions = [self.predict(row['user_id'], row['item_id']) 
                      for _, row in test_df.iterrows()]
        
        rmse = np.sqrt(mean_squared_error(test_df['rating'], predictions))
        mae = mean_absolute_error(test_df['rating'], predictions)
        
        return {'RMSE': rmse, 'MAE': mae}
    
class UserBasedCF:
    """
    Collaborative Filtering baseado em similaridade de usuários.
    
    CORREÇÕES IMPLEMENTADAS:
    - Validação de valores NaN/Inf em todas as predições
    - Fallback robusto para casos extremos
    - Logging de predições problemáticas
    - Normalização correta dos ratings
    """
    
    def __init__(self, k_neighbors: int = 20, min_neighbors: int = 3):
        self.k_neighbors = k_neighbors
        self.min_neighbors = min_neighbors  # Mínimo de vizinhos para confiar na predição
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_mean = None
        self.global_mean = None
        self.problem_predictions = []  # Para debug
    
    def fit(self, ratings_df: pd.DataFrame) -> 'UserBasedCF':
        """Treina o modelo com validações adicionais"""
        
        print("   Criando matriz usuário-item...")
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating'
        )
        
        # Calcular médias (IMPORTANTE: com validação)
        self.user_mean = self.user_item_matrix.mean(axis=1)
        self.global_mean = ratings_df['rating'].mean()
        
        # Verificar se há NaN nas médias
        if self.user_mean.isna().any():
            print(f"   ⚠️ {self.user_mean.isna().sum()} usuários sem média válida")
            self.user_mean = self.user_mean.fillna(self.global_mean)
        
        # Normalizar ratings (subtrair média do usuário)
        user_item_normalized = self.user_item_matrix.sub(self.user_mean, axis=0).fillna(0)
        
        print("   Calculando similaridade entre usuários...")
        self.user_similarity = cosine_similarity(user_item_normalized)
        
        # Verificar se há NaN/Inf na matriz de similaridade
        if np.isnan(self.user_similarity).any() or np.isinf(self.user_similarity).any():
            print("   ⚠️ Similaridades inválidas detectadas, corrigindo...")
            self.user_similarity = np.nan_to_num(self.user_similarity, nan=0.0, posinf=1.0, neginf=0.0)
        
        print(f"   ✅ Matriz: {self.user_item_matrix.shape} | K-vizinhos: {self.k_neighbors}")
        print(f"   📊 Média global: {self.global_mean:.2f}")
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Prediz rating com validações robustas"""
        
        # CASO 1: Usuário ou item não existe → média global
        if user_id not in self.user_item_matrix.index:
            return self.global_mean
        
        if item_id not in self.user_item_matrix.columns:
            return self.user_mean[user_id]
        
        # CASO 2: Usuário já avaliou o item → retornar rating conhecido
        user_rating = self.user_item_matrix.loc[user_id, item_id]
        if not np.isnan(user_rating):
            return float(user_rating)
        
        # CASO 3: Fazer predição baseada em vizinhos
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            similarities = self.user_similarity[user_idx]
            
            # Pegar ratings dos vizinhos para este item
            item_ratings = self.user_item_matrix[item_id]
            rated_users_mask = ~item_ratings.isna()
            
            # Se ninguém avaliou este item → média do usuário
            if rated_users_mask.sum() == 0:
                return self.user_mean[user_id]
            
            neighbor_ratings = item_ratings[rated_users_mask]
            neighbor_similarities = similarities[rated_users_mask]
            
            # Remover similaridades negativas ou zero
            positive_mask = neighbor_similarities > 0
            if positive_mask.sum() < self.min_neighbors:
                # Poucos vizinhos confiáveis → usar média do usuário
                return self.user_mean[user_id]
            
            neighbor_ratings = neighbor_ratings[positive_mask]
            neighbor_similarities = neighbor_similarities[positive_mask]
            
            # Pegar top-k vizinhos
            if len(neighbor_similarities) > self.k_neighbors:
                top_k_idx = np.argsort(neighbor_similarities)[-self.k_neighbors:]
                neighbor_ratings = neighbor_ratings.iloc[top_k_idx]
                neighbor_similarities = neighbor_similarities[top_k_idx]
            
            # Calcular predição ponderada
            total_similarity = neighbor_similarities.sum()
            
            if total_similarity == 0:
                prediction = self.user_mean[user_id]
            else:
                prediction = np.average(neighbor_ratings, weights=neighbor_similarities)
            
            # VALIDAÇÃO FINAL: Verificar se predição é válida
            if np.isnan(prediction) or np.isinf(prediction):
                self.problem_predictions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'prediction': prediction,
                    'fallback': self.user_mean[user_id]
                })
                return self.user_mean[user_id]
            
            # Limitar predição ao range válido (ex: 1-5 para MovieLens)
            min_rating = 1.0
            max_rating = 5.0
            prediction = np.clip(prediction, min_rating, max_rating)
            
            return float(prediction)
        
        except Exception as e:
            # Qualquer erro → fallback seguro
            print(f"   ⚠️ Erro na predição: {e}")
            return self.user_mean[user_id]
    
    def recommend(self, user_id: int, n: int = 5) -> list[int]:
        """Retorna top-N recomendações"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings.isna()].index
        
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in predictions[:n]]
    
    def evaluate(self, test_df: pd.DataFrame, verbose: bool = True) -> dict:
        """Avalia modelo com diagnóstico detalhado"""
        
        if verbose:
            print("   Avaliando modelo no conjunto de teste...")
        
        predictions = []
        actuals = []
        invalid_count = 0
        
        for idx, row in test_df.iterrows():
            pred = self.predict(row['user_id'], row['item_id'])
            
            # Verificar validade da predição
            if np.isnan(pred) or np.isinf(pred):
                invalid_count += 1
                pred = self.global_mean
            
            predictions.append(pred)
            actuals.append(row['rating'])
            
            if verbose and (idx + 1) % 5000 == 0:
                print(f"      Progresso: {idx+1}/{len(test_df)} predições")
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        # Diagnóstico adicional
        predictions_array = np.array(predictions)
        
        if verbose:
            print(f"\n   🔍 Diagnóstico das predições:")
            print(f"      Predições inválidas: {invalid_count} ({invalid_count/len(predictions)*100:.2f}%)")
            print(f"      Predições mínima: {predictions_array.min():.2f}")
            print(f"      Predições máxima: {predictions_array.max():.2f}")
            print(f"      Predições média: {predictions_array.mean():.2f}")
            print(f"      Predições std: {predictions_array.std():.2f}")
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'invalid_predictions': invalid_count,
            'predictions_stats': {
                'min': float(predictions_array.min()),
                'max': float(predictions_array.max()),
                'mean': float(predictions_array.mean()),
                'std': float(predictions_array.std())
            }
        }
    
class ItemBasedCF:
    """Collaborative Filtering baseado em similaridade de itens"""
    
    def __init__(self, k_neighbors=20):
        self.k_neighbors = k_neighbors
        self.item_similarity = None
        self.user_item_matrix = None
    
    def fit(self, ratings_df):
        print("   Criando matriz item-usuário...")
        # Transpor para ter itens nas linhas
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating'
        )
        
        item_user_matrix = self.user_item_matrix.T.fillna(0)
        
        print("   Calculando similaridade entre itens...")
        self.item_similarity = cosine_similarity(item_user_matrix)
        
        print(f"   ✅ {len(item_user_matrix)} itens processados")
        return self
    
    def predict(self, user_id, item_id):
        if user_id not in self.user_item_matrix.index or \
           item_id not in self.user_item_matrix.columns:
            return self.user_item_matrix.mean().mean()
        
        # Itens que o usuário avaliou
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[~user_ratings.isna()]
        
        if len(rated_items) == 0:
            return self.user_item_matrix.mean().mean()
        
        # Similaridades do item target com itens avaliados
        item_idx = self.user_item_matrix.columns.get_loc(item_id)
        similarities = []
        ratings = []
        
        for rated_item_id in rated_items.index:
            rated_item_idx = self.user_item_matrix.columns.get_loc(rated_item_id)
            sim = self.item_similarity[item_idx, rated_item_idx]
            
            if sim > 0:
                similarities.append(sim)
                ratings.append(rated_items[rated_item_id])
        
        if len(similarities) == 0:
            return self.user_item_matrix[item_id].mean()
        
        # Top-k itens similares
        if len(similarities) > self.k_neighbors:
            top_k_idx = np.argsort(similarities)[-self.k_neighbors:]
            similarities = [similarities[i] for i in top_k_idx]
            ratings = [ratings[i] for i in top_k_idx]
        
        # Predição ponderada
        prediction = np.average(ratings, weights=similarities)
        return prediction
    
    def evaluate(self, test_df):
        print("   Avaliando modelo item-based...")
        predictions = []
        actuals = []
        
        for idx, row in test_df.iterrows():
            pred = self.predict(row['user_id'], row['item_id'])
            predictions.append(pred)
            actuals.append(row['rating'])
            
            if (idx + 1) % 5000 == 0:
                print(f"      Progresso: {idx+1}/{len(test_df)}")
        
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        
        return {'RMSE': rmse, 'MAE': mae}