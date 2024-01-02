import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.1f' % x)
pd.set_option('display.max_rows', 500)


reviews = pd.read_csv("dataset/lastDfs/comments.csv")
products = pd.read_csv("dataset/lastDfs/products.csv")

app_name = products["app_name"]
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

title_vectors = model.encode(app_name)
vector_dimension = title_vectors.shape[1]

index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(title_vectors)
index.add(title_vectors)

faiss.write_index(index, 'path_to_your_index_file.index')




app_name_index = faiss.read_index('game_names_faiss.index')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

search_text = "Batman: Arkham Asylum Game of the Year Edition"
search_vector = model.encode(search_text)
_vector = np.array([search_vector])
faiss.normalize_L2(_vector)

k = app_name_index.ntotal
distances, ann = app_name_index.search(_vector, k=k)

results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})

products.loc[ann[0], :].head(50).iloc[1:50,:]