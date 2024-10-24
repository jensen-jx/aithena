from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.base import logger, IndexManagement, MILVUS_ID_FIELD, DEFAULT_BATCH_SIZE

from typing import Any, Dict, List, Optional, Union

from llama_index.vector_stores.milvus.utils import (
    get_default_sparse_embedding_function,
    BaseSparseEmbeddingFunction,
)
from llama_index.core.vector_stores.utils import (
    DEFAULT_DOC_ID_KEY,
    DEFAULT_EMBEDDING_KEY,
)
from pymilvus import Collection, MilvusClient, DataType
from pymilvus.client.types import LoadState
'''
Custom Class:
1. check if db is present. if not, create one
'''
class CustomMilvusVectorStore(MilvusVectorStore):
    def __init__(
        self,
        uri: str = "./milvus_llamaindex.db",
        token: str = "",
        collection_name: str = "llamacollection",
        dim: Optional[int] = None,
        embedding_field: str = DEFAULT_EMBEDDING_KEY,
        doc_id_field: str = DEFAULT_DOC_ID_KEY,
        similarity_metric: str = "IP",
        consistency_level: str = "Session",
        overwrite: bool = False,
        text_key: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        index_config: Optional[dict] = None,
        search_config: Optional[dict] = None,
        collection_properties: Optional[dict] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        enable_sparse: bool = False,
        sparse_embedding_function: Optional[BaseSparseEmbeddingFunction] = None,
        hybrid_ranker: str = "RRFRanker",
        hybrid_ranker_params: dict = {},
        index_management: IndexManagement = IndexManagement.CREATE_IF_NOT_EXISTS,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        super(MilvusVectorStore, self).__init__(
            collection_name=collection_name,
            dim=dim,
            embedding_field=embedding_field,
            doc_id_field=doc_id_field,
            consistency_level=consistency_level,
            overwrite=overwrite,
            text_key=text_key,
            output_fields=output_fields or [],
            index_config=index_config if index_config else {},
            search_config=search_config if search_config else {},
            collection_properties=collection_properties,
            batch_size=batch_size,
            enable_sparse=enable_sparse,
            sparse_embedding_function=sparse_embedding_function,
            hybrid_ranker=hybrid_ranker,
            hybrid_ranker_params=hybrid_ranker_params,
            index_management=index_management,
        )

        # Select the similarity metric
        similarity_metrics_map = {
            "ip": "IP",
            "l2": "L2",
            "euclidean": "L2",
            "cosine": "COSINE",
        }
        self.similarity_metric = similarity_metrics_map.get(
            similarity_metric.lower(), "L2"
        )
        # Connect to Milvus instance
        if kwargs.get("db_name") is not None:
            self._milvusclient = MilvusClient(
                uri=uri,
                token=token                 
            )
            dbs = self._milvusclient.list_databases()
            if kwargs['db_name'] not in dbs:
                self._milvusclient.create_database(kwargs['db_name'])
         
        self._milvusclient = MilvusClient(
            uri=uri,
            token=token,
            **kwargs,  # pass additional arguments such as server_pem_path
        )
        # Delete previous collection if overwriting
        if overwrite and collection_name in self.client.list_collections():
            self._milvusclient.drop_collection(collection_name)

        # Create the collection if it does not exist
        if collection_name not in self.client.list_collections():
            if dim is None:
                raise ValueError("Dim argument required for collection creation.")
            if self.enable_sparse is False:
                self._milvusclient.create_collection(
                    collection_name=collection_name,
                    dimension=dim,
                    primary_field_name=MILVUS_ID_FIELD,
                    vector_field_name=embedding_field,
                    id_type="string",
                    metric_type=self.similarity_metric,
                    max_length=65_535,
                    consistency_level=consistency_level,
                )
            else:
                try:
                    _ = DataType.SPARSE_FLOAT_VECTOR
                except Exception as e:
                    logger.error(
                        "Hybrid retrieval is only supported in Milvus 2.4.0 or later."
                    )
                    raise NotImplementedError(
                        "Hybrid retrieval requires Milvus 2.4.0 or later."
                    ) from e
                self._create_hybrid_index(collection_name)

        self._collection = Collection(collection_name, using=self._milvusclient._using)
        self._create_index_if_required()

        # Set properties
        if collection_properties:
            if self._milvusclient.get_load_state(collection_name) == LoadState.Loaded:
                self._collection.release()
                self._collection.set_properties(properties=collection_properties)
                self._collection.load()
            else:
                self._collection.set_properties(properties=collection_properties)

        self.enable_sparse = enable_sparse
        if self.enable_sparse is True and sparse_embedding_function is None:
            logger.warning("Sparse embedding function is not provided, using default.")
            self.sparse_embedding_function = get_default_sparse_embedding_function()
        elif self.enable_sparse is True and sparse_embedding_function is not None:
            self.sparse_embedding_function = sparse_embedding_function
        else:
            pass

        logger.debug(f"Successfully created a new collection: {self.collection_name}")