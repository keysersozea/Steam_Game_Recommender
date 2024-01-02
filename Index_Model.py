import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
def Index_Model(product_name):
    app_name_index = faiss.read_index('game_names_faiss.index')
    search_text = product_name
    search_vector = model.encode(search_text)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)

    k = app_name_index.ntotal
    distances, ann = app_name_index.search(_vector, k=k)
    # Bu model product dataframe'indeki app_name değişkeni listeye çevirilerek eğitildi
    # ann listesi en yakın productların indexlerini tutuyor
    # ann'i product df'i ile kullanmak gerekiyor
    return ann
