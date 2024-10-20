from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

def get_embeddings(path: str, model_name) -> HuggingFaceEmbedding:
    path = os.path.join(path, model_name)
    return HuggingFaceEmbedding(model_name=path)

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


def get_langchain_embeddings(path: str, model_name) -> HuggingFaceEmbeddings:
    path = os.path.join(path, model_name)
    return HuggingFaceEmbeddings(model_name=path)