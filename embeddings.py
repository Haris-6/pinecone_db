import numpy as np
from sentence_transformers import SentenceTransformer #his class is used to convert text into vector embeddings, which capture the meaning of text numerically.
from config import EMBEDDING_MODEL

def init_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def embed_texts(model, texts):
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False) #Generates embeddings for the given texts.and convert_to_numpy=True: Converts the output from a PyTorch/TensorFlow tensor to a NumPy array, which is easier to store and manipulate.
    return emb.astype(np.float32) #Converts the embeddings to 32-bit floating point
