from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from typing import List
import chromadb
import qdrant_client
import os
def get_chroma_vectorstore(db_dir: str, collection_name: str, db_name: str) -> ChromaVectorStore:
    path = os.path.join(db_dir, db_name)
    db = chromadb.PersistentClient(path=path)
    collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return vector_store

 