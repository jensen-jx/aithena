from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os
def get_vectorstore(db_dir: str, collection_name: str, db_name: str) -> ChromaVectorStore:
    path = os.path.join(db_dir, db_name)
    db = chromadb.PersistentClient(path=path)
    collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return vector_store
