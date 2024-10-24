from llama_index.core.indices import DocumentSummaryIndex
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Union, cast
 
from llama_index.core.indices.utils import embed_nodes
from llama_index.core.base.response.schema import Response
from llama_index.core.data_structs.document_summary import IndexDocumentSummary
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import (
    BaseNode,
    NodeRelationship,
    NodeWithScore,
    RelatedNodeInfo,
    TextNode,
)
from llama_index.core.utils import get_tqdm_iterable

from common_lib.utils.async_utils import async_run

logger = logging.getLogger(__name__)
'''
Custom Class
1. Add async function
'''
class CustomDocumentSummaryIndex(DocumentSummaryIndex):

    async def _async_add_nodes_to_index(
        self,
        index_struct: IndexDocumentSummary,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> None:
        """Add nodes to index."""
        doc_id_to_nodes = defaultdict(list)
        for node in nodes:
            if node.ref_doc_id is None:
                raise ValueError(
                    "ref_doc_id of node cannot be None when building a document "
                    "summary index"
                )
            doc_id_to_nodes[node.ref_doc_id].append(node)

        summary_node_dict = {}
        items = doc_id_to_nodes.items()
        iterable_with_progress = get_tqdm_iterable(
            items, show_progress, "Summarizing documents"
        )
 
        tasks = [self._response_synthesizer.asynthesize(
                        query=self._summary_query,
                        nodes=[NodeWithScore(node=n) for n in nodes],
                    ) 
                for _, nodes in iterable_with_progress]
        summary_responses = await async_run(tasks, 15)
        idx = 0
        for doc_id, nodes in iterable_with_progress:
            print(f"current doc id: {doc_id}")
            # get the summary for each doc_id
            summary_response = summary_responses[idx]
            idx +=1
            summary_response = cast(Response, summary_response)
            docid_first_node = doc_id_to_nodes.get(doc_id, [TextNode()])[0]
            summary_node_dict[doc_id] = TextNode(
                text=summary_response.response,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc_id)
                },
                metadata=docid_first_node.metadata,
                excluded_embed_metadata_keys=docid_first_node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=docid_first_node.excluded_llm_metadata_keys,
            )
            self.docstore.add_documents([summary_node_dict[doc_id]])
            logger.info(
                f"> Generated summary for doc {doc_id}: " f"{summary_response.response}"
            )

        for doc_id, nodes in doc_id_to_nodes.items():
            index_struct.add_summary_and_nodes(summary_node_dict[doc_id], nodes)

        if self._embed_summaries:
            summary_nodes = list(summary_node_dict.values())
            id_to_embed_map = embed_nodes(
                summary_nodes, self._embed_model, show_progress=show_progress
            )

            summary_nodes_with_embedding = []
            for node in summary_nodes:
                node_with_embedding = node.model_copy()
                node_with_embedding.embedding = id_to_embed_map[node.node_id]
                summary_nodes_with_embedding.append(node_with_embedding)
            self._vector_store.add(summary_nodes_with_embedding)