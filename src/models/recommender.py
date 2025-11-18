"""
Módulo de Produção - Sistema de Recomendação
Versão limpa e otimizada para deploy
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from src.models.collaborative_filtering import BaselineRecommender, UserBasedCF, ItemBasedCF

class RecommenderSystem:
    """
    Sistema de recomendação pronto para produção.
    Utiliza o melhor modelo (Item-Based CF) treinado.
    
    Uso:
        recommender = RecommenderSystem()
        recommender.load_model('models/item_based_cf_model.pkl')
        recommendations = recommender.recommend(user_id=1, n=10)

    Métodos:
        * load_model(model_path): 'RecommenderSystem' -> Carrega modelo treinado.
        * recommend(user_id, n, exclude_rated): List[Dict] -> Gera recomendações personalizadas.
        * predict(user_id, item_id): float -> Prediz rating para um par usuário-item.
        * similar_items(item_id, n): List[Dict] -> Encontra itens similares (content-based).
        * get_user_profile(user_id): Dict -> Retorna perfil do usuário com estatísticas.
    """
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.metadata = {}
    
    def load_model(self, model_path: str) -> 'RecommenderSystem':
        """
        Carrega modelo treinado.
        
        Parâmetros:
            * model_path: str -> Caminho para o arquivo do modelo.
        
        Retorna:
            * self: RecommenderSystem -> Instância com modelo carregado.
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Detectar tipo de modelo
        self.model_type = self.model.__class__.__name__
        
        # Extrair metadados
        self.metadata = {
            'model_type': self.model_type,
            'num_users': len(self.model.user_item_matrix.index),
            'num_items': len(self.model.user_item_matrix.columns),
            'sparsity': self._calculate_sparsity()
        }
        
        print(f"✅ Modelo carregado: {self.model_type}")
        print(f"   Usuários: {self.metadata['num_users']:,}")
        print(f"   Itens: {self.metadata['num_items']:,}")
        print(f"   Esparsidade: {self.metadata['sparsity']:.2%}")
        
        return self
    
    def recommend(
        self, 
        user_id: int, 
        n: int = 10,
        exclude_rated: bool = True
    ) -> List[Dict]:
        """
        Gera recomendações personalizadas.
        
        Parâmetros:
            * user_id: int -> ID do usuário.
            * n: int -> Número de recomendações.
            * exclude_rated: bool -> Excluir itens já avaliados?

        Retorna:
            * List[Dict] -> Lista de recomendações com scores e confiança.
        """

        if self.model is None:
            raise ValueError("Modelo não carregado. Use load_model() primeiro.")
        
        # Verificar se usuário existe
        if user_id not in self.model.user_item_matrix.index:
            return self._cold_start_recommendations(n)
        
        # Gerar recomendações
        if hasattr(self.model, 'recommend'):
            item_ids = self.model.recommend(user_id, n)
        else:
            item_ids = self._fallback_recommend(user_id, n)
        
        # Calcular scores estimados
        recommendations = []
        for item_id in item_ids:
            score = self.model.predict(user_id, item_id)
            recommendations.append({
                'item_id': int(item_id),
                'estimated_rating': float(score),
                'confidence': self._calculate_confidence(user_id, item_id)
            })
        
        return recommendations
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Prediz rating para um par usuário-item.
        
        Parâmetros:
            * user_id: int -> ID do usuário.
            * item_id: int -> ID do item.
        Retorna:
            * float -> Rating predito
        """
        if self.model is None:
            raise ValueError("Modelo não carregado.")
        
        return self.model.predict(user_id, item_id)
    
    def similar_items(self, item_id: int, n: int = 10) -> List[Dict]:
        """
        Encontra itens similares (content-based).
        
        Parâmetros:
            * item_id: int -> ID do item.
            * n: int -> Número de itens similares.
        Retorna:
            * List[Dict] -> Itens similares com scores de similaridade.
        """

        if self.model_type != 'ItemBasedCF':
            raise NotImplementedError("Disponível apenas para Item-Based CF")
        
        if item_id not in self.model.user_item_matrix.columns:
            return []
        
        item_idx = self.model.user_item_matrix.columns.get_loc(item_id)
        similarities = self.model.item_similarity[item_idx]
        
        # Top N mais similares (excluindo o próprio item)
        similar_indices = similarities.argsort()[-n-1:-1][::-1]
        
        results = []
        for idx in similar_indices:
            similar_item_id = self.model.user_item_matrix.columns[idx]
            results.append({
                'item_id': int(similar_item_id),
                'similarity_score': float(similarities[idx])
            })
        
        return results
    
    def get_user_profile(self, user_id: int) -> Dict:
        """
        Retorna perfil do usuário com estatísticas.
        Parâmetros:
            * user_id: int -> ID do usuário.
        Retorna:
            * Dict -> Perfil do usuário com estatísticas.
        """

        if user_id not in self.model.user_item_matrix.index:
            return {'error': 'Usuário não encontrado'}
        
        user_ratings = self.model.user_item_matrix.loc[user_id]
        rated_items = user_ratings.dropna()
        
        return {
            'user_id': int(user_id),
            'num_ratings': int(len(rated_items)),
            'avg_rating': float(rated_items.mean()),
            'rating_std': float(rated_items.std()),
            'favorite_items': rated_items.nlargest(5).index.tolist(),
            'least_favorite_items': rated_items.nsmallest(5).index.tolist()
        }
    
    def get_trending(self, n: int = 10, min_ratings: int = 20) -> List[Dict]:
        """
        Retorna itens mais populares (trending).
        
        Parâmetros:
            * n: int -> Número de itens trending.
            * min_ratings: int -> Mínimo de avaliações para considerar.
        Retorna:
            * List[Dict] -> Itens trending com estatísticas.
        """

        item_stats = self.model.user_item_matrix.apply(
            lambda col: {
                'count': col.notna().sum(),
                'mean': col.mean()
            }
        )
        
        trending = []
        for item_id, stats in item_stats.items():
            if stats['count'] >= min_ratings:
                trending.append({
                    'item_id': int(item_id),
                    'avg_rating': float(stats['mean']),
                    'num_ratings': int(stats['count']),
                    'popularity_score': float(stats['mean'] * np.log(stats['count']))
                })
        
        # Ordenar por popularidade
        trending.sort(key=lambda x: x['popularity_score'], reverse=True)
        return trending[:n]
    
    def _cold_start_recommendations(self, n: int) -> List[Dict]:
        """
        Recomendações para usuários novos (cold start).
        Parâmetros:
            * n: int -> Número de recomendações.
        Retorna:
            * List[Dict] -> Recomendações baseadas em trending.
        """

        trending = self.get_trending(n, min_ratings=50)
        return [{
            'item_id': item['item_id'],
            'estimated_rating': item['avg_rating'],
            'confidence': 0.5  # Confiança baixa para cold start
        } for item in trending]
    
    def _calculate_confidence(self, user_id: int, item_id: int) -> float:
        """
        Calcula confiança da recomendação (0-1).
        Baseado no número de avaliações do usuário e popularidade do item.
        Parâmetros:
            * user_id: int -> ID do usuário.
            * item_id: int -> ID do item.
        Retorna:
            * float -> Confiança (0-1).
        """
        user_ratings_count = self.model.user_item_matrix.loc[user_id].notna().sum()
        item_ratings_count = self.model.user_item_matrix[item_id].notna().sum()
        
        # Normalizar (mais avaliações = mais confiança)
        user_confidence = min(user_ratings_count / 50, 1.0)
        item_confidence = min(item_ratings_count / 100, 1.0)
        
        return float((user_confidence + item_confidence) / 2)
    
    def _calculate_sparsity(self) -> float:
        """
        Calcula esparsidade da matriz.
        Retorna:
            * float -> Esparsidade (0-1).
    """
        total_cells = self.model.user_item_matrix.size
        filled_cells = self.model.user_item_matrix.notna().sum().sum()
        return 1 - (filled_cells / total_cells)
    
    def _fallback_recommend(self, user_id: int, n: int) -> List[int]:
        """
        Fallback se modelo não tiver método recommend().
        Parâmetros:
            * user_id: int -> ID do usuário.
            * n: int -> Número de recomendações.
        Retorna:
            * List[int] -> IDs dos itens recomendados.
        """

        user_ratings = self.model.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings.isna()].index.tolist()
        
        predictions = [
            (item_id, self.model.predict(user_id, item_id))
            for item_id in unrated_items[:min(len(unrated_items), n*3)]
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return [item_id for item_id, _ in predictions[:n]]
    
    def health_check(self) -> Dict:
        """
        Verifica saúde do sistema.
        Retorna:
            * Dict -> Status do sistema e metadados.
        """

        return {
            'status': 'healthy' if self.model is not None else 'not_loaded',
            'model_loaded': self.model is not None,
            'model_type': self.model_type,
            'metadata': self.metadata
        }


# =============================================================================
# EXEMPLO DE USO
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🚀 TESTANDO SISTEMA DE RECOMENDAÇÃO EM PRODUÇÃO")
    print("=" * 80)
    
    # 1. Inicializar sistema
    recommender = RecommenderSystem()
    
    # 2. Carregar melhor modelo
    recommender.load_model('models/item_based_cf_model.pkl')
    
    # 3. Testar recomendações
    print("\n📍 Teste 1: Recomendações personalizadas")
    user_id = 1
    recommendations = recommender.recommend(user_id, n=10)
    
    print(f"\nTop 10 recomendações para usuário {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. Item {rec['item_id']:4d} | "
              f"Rating: {rec['estimated_rating']:.2f} | "
              f"Confiança: {rec['confidence']:.0%}")
    
    # 4. Testar itens similares
    print("\n📍 Teste 2: Itens similares")
    item_id = recommendations[0]['item_id']
    similar = recommender.similar_items(item_id, n=5)
    
    print(f"\nItens similares a {item_id}:")
    for i, item in enumerate(similar, 1):
        print(f"{i}. Item {item['item_id']:4d} | "
              f"Similaridade: {item['similarity_score']:.3f}")
    
    # 5. Testar perfil do usuário
    print("\n📍 Teste 3: Perfil do usuário")
    profile = recommender.get_user_profile(user_id)
    print(f"\nPerfil do usuário {user_id}:")
    print(f"   Avaliações: {profile['num_ratings']}")
    print(f"   Média: {profile['avg_rating']:.2f}")
    print(f"   Top 5 favoritos: {profile['favorite_items']}")
    
    # 6. Testar trending
    print("\n📍 Teste 4: Itens em alta")
    trending = recommender.get_trending(n=5)
    print("\nTop 5 trending:")
    for i, item in enumerate(trending, 1):
        print(f"{i}. Item {item['item_id']:4d} | "
              f"Rating: {item['avg_rating']:.2f} | "
              f"Avaliações: {item['num_ratings']}")
    
    # 7. Health check
    print("\n📍 Teste 5: Health Check")
    health = recommender.health_check()
    print(f"\nStatus: {health['status']}")
    print(f"Modelo: {health['model_type']}")
    print(f"Usuários: {health['metadata']['num_users']:,}")
    
    print("\n" + "=" * 80)
    print("✅ TODOS OS TESTES CONCLUÍDOS!")
    print("=" * 80)