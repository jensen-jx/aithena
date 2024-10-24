from common_lib.utils.loader import load_pdf, load_pdf_API
from common_lib.utils.embedding_utils import get_embeddings, get_langchain_embeddings
from common_lib.utils.llm_utils import get_llm
from common_lib.utils.async_utils import async_run
from common_lib.utils.load_config import load_config
from common_lib.custom_models.custom_document_summary import CustomDocumentSummaryIndex
from common_lib.custom_models.custom_milvus_vectorstore import CustomMilvusVectorStore

from typing import List, Dict
from langchain_experimental.text_splitter import SemanticChunker
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core import Document, StorageContext, DocumentSummaryIndex
from llama_index.core.ingestion import run_transformations
from llama_index.core.settings import Settings
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.vector_stores.milvus import MilvusVectorStore

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
import itertools
import asyncio
from pathlib import Path
class IngestionSvc:
    def __init__(self, db_dir:str, llm_config: Dict[str, str], vectorstore_config: Dict[str, str], embedding_config: Dict[str, str]):
        load_dotenv()

        self.embeddings = get_embeddings(**embedding_config)
        self.semantic_chunker = SemanticChunker(embeddings=get_langchain_embeddings(**embedding_config))

        self.extraction_types = ["cases", "case_summaries"]

        self.mongo_uri = os.getenv('MONGO_URI')
        self.milvus_uri = os.getenv('MILVUS_URI')

        self.ingested_db = AsyncIOMotorClient(self.mongo_uri).ingested_db.ingested

        self.vector_stores = {}
        self.storage_contexts = {}
        self.indices = {}
        self.llm = get_llm(**llm_config)

        self.init_datastructs(vectorstore_config)

    
    def init_datastructs(self, vectorstore_config: Dict[str, str]):    
        indices_classes = {
            "cases": VectorStoreIndex,
            "case_summaries": CustomDocumentSummaryIndex
        }
        for name in self.extraction_types:
            namespace = name
            vectorstore = None
            storagecontext = None
             
            vectorstore = CustomMilvusVectorStore(uri=self.milvus_uri, **vectorstore_config[name])            
            docstore = MongoDocumentStore.from_uri(self.mongo_uri, db_name="docstore", namespace=namespace)
            index_store = MongoIndexStore.from_uri(self.mongo_uri, db_name="indexstore", namespace=namespace)
            storagecontext = StorageContext.from_defaults(vector_store=vectorstore, docstore=docstore, index_store=index_store)
            
            index_structs = storagecontext.index_store.index_structs()
            nodes, index_struct = None, None
            if len(index_structs) == 1:
                index_struct = index_structs[0]
            elif len(index_structs) == 0:
                nodes = []
            else:
                assert len(index_structs) <= 1, "Total number of existing index structs has exceeded 1"

            self.vector_stores[name] = storagecontext.vector_store
            self.storage_contexts[name] = storagecontext

            self.indices[name] = indices_classes[name](
                                    nodes=nodes, llm = self.llm, index_struct=index_struct, storage_context=storagecontext, embed_model=self.embeddings
                                )
      
         

    def process_pdf(self, file_path: str) -> List[Document]:
        
        file_name = Path(file_path).name
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

    async def add_document_to_indices(self, path: str) -> None:
        documents  = self.process_pdf(path)
        for name in self.extraction_types:
            print(f"Creating Index for {name}")
            index = self.indices[name]
            docstore = index.docstore
            callback_manager = Settings.callback_manager
            transformations = Settings.transformations

            with callback_manager.as_trace("index_construction"):
                for doc in documents:
                    docstore.set_document_hash(doc.get_doc_id(), doc.hash)

                nodes = run_transformations(
                    documents,  # type: ignore
                    transformations,
                    show_progress=True,
                )
                
                docstore.add_documents(nodes, allow_update=True)  
                index_struct = index.storage_context.index_store.get_index_struct()           
                await index._async_add_nodes_to_index(nodes=nodes, index_struct=index_struct)

                index.storage_context.index_store.add_index_struct(index_struct)
        
        record = {"file": Path(path).name, "success": True}
        await self.ingested_db.insert_one(record)
        

    async def add_documents_to_indices(self, paths: List[str]) -> bool:
        print("Loading PDFs")
        tasks = []

        for path in paths:
            if Path(path).suffix != ".pdf": continue
            elif await self.ingested_db.find_one({"file": Path(path).name}) is not None: 
                print(f"{path} has been ingested before. This file will be ignored.")
                continue
            await self.add_document_to_indices(path)
     