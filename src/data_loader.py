import os
import pandas as pd
from pathlib import Path

def download_movielens() -> pd.DataFrame:
    """Baixa dataset MovieLens 100K"""
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    
    import zipfile, requests, io
    
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall("data/raw")
    
    # Carregue os dados
    ratings = pd.read_csv(
        'data/raw/ml-100k/u.data',
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    return ratings

if __name__ == "__main__":
    df = download_movielens()
    print(f"✅ {len(df)} avaliações carregadas")
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/ratings.csv', index=False)