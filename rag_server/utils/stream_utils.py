from llama_index.core.base.response.schema import StreamingResponse, AsyncStreamingResponse
from typing import Union
from rag_server.event_model.event import Event

import json
def convert_sse(obj: str | dict):
    """Convert the given object (or string) to a Server-Sent Event (SSE) event"""
    # print(obj)
    return "data: {}\n\n".format(json.dumps(obj))

def stream_generator(id: str, response: StreamingResponse):
    text = ""
    for token in response.response_gen:
        text += token

        yield convert_sse(token)

    sources = {}
    print(len(response.source_nodes))
    for source_node in response.source_nodes:
        source_text = source_node.node.get_content()
        source = source_node.node.metadata.get('file_name')
        if source is None:
            source = source_node.node.id_
        sources[source] = source_text
        if source_text is not "":
            yield convert_sse("\n\n---\n\n")
            yield convert_sse("\n\n**Source:** \t" + source)
            yield convert_sse("\n\n**Source text:** \t" + source_text)

    #yield convert_sse(sources)