from common_lib.utils.embedding_utils import get_embeddings
from common_lib.utils.vectorstore_utils import get_vectorstore
from common_lib.utils.llm_utils import get_llm
from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.core.indices.document_summary import DocumentSummaryIndex, DocumentSummaryIndexEmbeddingRetriever
from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank

from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore

from typing import Dict, List, Any

import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class QueryEngineController:
    def __init__(self, config: Dict[str, Any]):
        logger.info("Loading QE Controller")
        load_dotenv()
        self.config = config
        db_dir = config['db_dir']
        self.extraction_types = ["cases", "case_summaries"]
        self.db_dirs = {name: os.path.join(db_dir, name) for name in self.extraction_types}  
        self.mongo_uri = os.getenv('MONGO_URI')
        self.embeddings = get_embeddings(**config['embeddings'])
        vectorstore_config = config['vectorstore']
        self.vector_stores = {}
        self.storage_contexts = {}
        self.indices = {}
        self.llm = get_llm(**self.config["llm"])

        self.init_datastructs(vectorstore_config)
        self.set_base_retrievers()
        self.set_node_processors()

    def init_datastructs(self, vectorstore_config: Dict[str, str]):    
        indices_classes = {
            "cases": VectorStoreIndex,
            "case_summaries": DocumentSummaryIndex
        }
        for name in self.extraction_types:
            namespace = name
            vectorstore = None
            storagecontext = None
            try:
                vectorstore = get_vectorstore(db_dir = self.db_dirs[name], **vectorstore_config[name])
            except:
                vectorstore = get_vectorstore(**vectorstore_config[name])
            
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
    
    def set_base_retrievers(self) -> List[BaseRetriever]:
        logger.info("Set base retrievers")
        self.base_retrievers = {}
        self.base_retrievers['cases'] = self.indices['cases'].as_retriever()
        self.base_retrievers['case_summaries'] = DocumentSummaryIndexEmbeddingRetriever(index=self.indices['case_summaries'], embed_model=self.embeddings, **self.config['retrievers']['summary_doc'])

    def set_node_processors(self) -> None:
        logger.info("Set node processors")
        path = self.config['postprocessors']['rerank'].pop("path")
        self.config['postprocessors']['rerank']['model'] = os.path.join(path, self.config['postprocessors']['rerank']['model'])
        rerank = SentenceTransformerRerank(**self.config['postprocessors']['rerank'])
        self.node_processors = []

    def get_query_engine(self) -> BaseQueryEngine:
        logger.info("Get query engine")
        self.fusion_retriever = QueryFusionRetriever(
                                    retrievers=list(self.base_retrievers.values()),
                                    llm=self.llm,
                                    **self.config['retrievers']['fusion']
                                )

        qe = RetrieverQueryEngine.from_args(
            retriever=self.fusion_retriever,
            llm=self.llm,
            response_mode="tree_summarize",
            node_postprocessors=self.node_processors,
            streaming=True,
            use_async=True
        )

        return qe

