from unstructured.partition.pdf import partition_pdf
from unstructured.partition.api import partition_via_api
from unstructured.documents.elements import Text, ElementMetadata, CompositeElement
from typing import List, Tuple
import os

def get_pdf_paths(directory: str) -> List[str]:
    paths = []
    for top, dirs, files in os.walk(directory):
        for file in files:
            paths.append(os.path.abspath(os.path.join(top, file)))
    return paths

def load_pdf(path: str) -> List[str]:
    elements = partition_pdf(path, strategy="hi_res", chunking_strategy="by_title", infer_table_structure=True)
    output = [ele.text for ele in elements if isinstance(ele, CompositeElement)]
    return output

def load_pdf_API(path: str) -> Tuple[List[str], List[ElementMetadata]]:
    print("Call UNSTRUCTURED API")
    elements = partition_via_api(
        filename=path, 
        api_url=os.environ["UNSTRUCTURED_API_URL"], 
        api_key="",
        strategy='hi_res',
        languages=["eng"],
        chunking_strategy="by_title"
    )
      
    output = [ele.text for ele in elements]
    return output
