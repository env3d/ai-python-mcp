# Suppress warnings and logging from Hugging Face and SentenceTransformers for cleaner output

import os
import warnings
import logging

# Hugging Face + Transformers verbosity
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# SentenceTransformers uses logging, not warnings
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Python warnings (e.g. fast/slow processor notice)
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
import faiss

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

verses = []

f = open('kjv.txt', encoding='utf-8-sig')
verses = f.readlines()
f.close()


def load_index(index_file='faiss_index.bin', embedding_dim=384):
    # Check if the FAISS index file exists
    if os.path.exists(index_file):
        # Load the FAISS index from the file
        index = faiss.read_index(index_file)
    else:
        # Read the verses from the file
        with open('kjv.txt', encoding='utf-8-sig') as f:
            print("Reading doc")
            verses = f.readlines()

        print("Creating index")
        # Encode the verses with batching for faster processing
        embeddings = model.encode(verses, normalize_embeddings=True, batch_size=320, show_progress_bar=True)

        # Create FAISS index with cosine similarity (IndexFlatIP on normalized vectors)
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)

        # Save the FAISS index to a file
        faiss.write_index(index, index_file)

    return index

index = load_index()

def search(query, top_k=3):
    query_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_emb, top_k)
    return [ verses[i] for i in I[0]]
