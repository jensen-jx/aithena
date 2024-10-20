from utils.embedding_utils import get_embeddings
from utils.vectorstore_utils import get_chroma_vectorstore
from utils.llm_utils import get_llm
from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.indices.managed.vectara import VectaraIndex, VectaraRetriever

from llama_index.core.indices.document_summary import DocumentSummaryIndex, DocumentSummaryIndexEmbeddingRetriever
from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank
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
        self.db_dir = config['db_dir']
        self.embeddings = get_embeddings(**config['embeddings'])
        config['vectorstore']['chroma']['db_dir'] = config['db_dir']

        self.chroma_vector_store = get_chroma_vectorstore(**config['vectorstore']['chroma'])
        self.storage_context = StorageContext.from_defaults(vector_store=self.chroma_vector_store, persist_dir=self.db_dir)
        self.llm = get_llm()

        self.set_indices()
        self.set_base_retrievers()
        self.set_node_processors()

    def set_indices(self) -> None:
        logger.info("Set indices")
        self.indices = {}
        self.indices['vectara'] = VectaraIndex(llm=self.llm, embed_model=self.embeddings)
        # self.indices['qdrant'] = VectorStoreIndex(vector_store=self.qdrant_vector_store, embed_model=self.embeddings)
        self.indices['summary'] = load_index_from_storage(self.storage_context, llm=self.llm, embed_model=self.embeddings)

    def set_base_retrievers(self) -> List[BaseRetriever]:
        logger.info("Set base retrievers")
        self.base_retrievers = {}
        # self.base_retrievers['qdrant'] = self.indices['qdrant'].as_retriever()
        self.base_retrievers['vectara'] = VectaraRetriever(self.indices['vectara'], **self.config['retrievers']['vectara'])
        self.base_retrievers['summary'] = DocumentSummaryIndexEmbeddingRetriever(index=self.indices['summary'], embed_model=self.embeddings, **self.config['retrievers']['summary_doc'])

    def set_node_processors(self) -> None:
        logger.info("Set node processors")
        path = self.config['postprocessors']['rerank'].pop("path")
        self.config['postprocessors']['rerank']['model'] = os.path.join(path, self.config['postprocessors']['rerank']['model'])
        rerank = SentenceTransformerRerank(**self.config['postprocessors']['rerank'])
        self.node_processors = [rerank]

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

