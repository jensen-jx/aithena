from ingestion.ingestion.utils.loader import load_pdf, load_pdf_API
from ingestion.ingestion.utils.embedding_utils import get_embeddings, get_langchain_embeddings
from ingestion.ingestion.utils.vectorstore_utils import get_vectorstore
from ingestion.ingestion.utils.llm_utils import get_llm
from ingestion.ingestion.utils.async_utils import concurrent_calls
from ingestion.ingestion.config.load_config import load_config
from typing import List, Dict
from langchain_experimental.text_splitter import SemanticChunker
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core import Document, StorageContext, DocumentSummaryIndex
from llama_index.indices.managed.vectara import VectaraIndex

import unstructured_client

from dotenv import load_dotenv
import os
import pathlib
import itertools
import asyncio
class IngestionSvc:
    def __init__(self, db_dir:str, vectorstore_config: Dict[str, str], embedding_config: Dict[str, str]):
        load_dotenv()

        self.embeddings = get_embeddings(**embedding_config)
        self.semantic_chunker = SemanticChunker(embeddings=get_langchain_embeddings(**embedding_config))
        self.db_dir = db_dir
        vectorstore_config['db_dir'] = db_dir
        self.vector_store = get_vectorstore(**vectorstore_config)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.llm = get_llm()

    def process_pdf(self, file_path: str) -> List[Document]:
        
        file_name = pathlib.Path(file_path).name
        print(f"Processing: {file_name}")

        # chunks = load_pdf(file_path)
        chunks = load_pdf_API(file_path)
        chunks = self.semantic_chunker.create_documents(chunks)
        chunks = [chunk.page_content for chunk in chunks]
        documents = []
        metadata = self.get_metadata(file_path, file_name)

        for chunk in chunks:
            doc = Document(text=chunk, metadata=metadata)
            documents.append(doc)
        return documents

    def get_metadata(self, file_path:str, file_name:str) -> Dict[str, str]:
        metadata = {'file_name' : file_name, 'file_path' : file_path}
        return metadata

    async def build_indices_from_documents(self, paths: List[str]) -> None:
        print("Loading PDFs")
        # loop = asyncio.get_event_loop()
        # paths = [path for path in paths if ".pdf" in path]
        # tasks = [loop.run_in_executor(None, self.process_pdf, path) for path in paths]
        # documents = await asyncio.gather(*tasks)
        # documents = await concurrent_calls(tasks, 2)
        documents = []
        for path in paths:
            documents.append(self.process_pdf(path))
        documents = list(itertools.chain.from_iterable(documents))
        # index = VectorStoreIndex.from_documents(
        #     documents, storage_context=self.storage_context, embed_model=self.embeddings
        # )

        print("Creating Document Summary Index")
        index = DocumentSummaryIndex.from_documents(
            documents,
            llm=self.llm,
            storage_context=self.storage_context,
            show_progress=True,
            embed_model=self.embeddings
        )

        index.storage_context.persist(self.db_dir)

        print("Creating vectara Index")
        VectaraIndex.from_documents(documents)
