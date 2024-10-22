from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

def get_embeddings(path: str, model_name:str, query_instruction:str = None, text_instruction:str = None) -> HuggingFaceEmbedding:
    path = os.path.join(path, model_name)
    return HuggingFaceEmbedding(model_name=path, trust_remote_code=True, query_instruction=query_instruction, text_instruction=text_instruction)

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings as LangchainHuggingFaceEmbeddings


def get_langchain_embeddings(path: str, model_name, query_instruction:str = None, text_instruction:str = None) -> LangchainHuggingFaceEmbeddings:
    path = os.path.join(path, model_name)
    model_kwargs={"trust_remote_code":True}
    return LangchainHuggingFaceEmbeddings(model_name=path, model_kwargs=model_kwargs)