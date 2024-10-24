from common_lib.utils.async_utils import async_run
from common_lib.utils.llm_utils import get_json_constraint_args
from common_lib.data_models.rhetorical_roles import RhetoricalRole
from common_lib.prompts import ROLES_EXTRACTION_SYSTEM_PROMPT, ROLES_EXTRACTION_USER_PROMPT
from typing import List, Dict, Any
from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import json


def craft_prompt(text: str) -> List[ChatMessage]:
    messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=ROLES_EXTRACTION_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=ROLES_EXTRACTION_USER_PROMPT.format(text=text)),
        ]
    return messages

def extract_roles(docs: List[str], llm: LLM) -> List[str]:
    results = []
    json_constraint_arg = get_json_constraint_args(llm, RhetoricalRole)
    for doc in docs:
        prompt = craft_prompt(doc)        
        result = llm.chat(prompt, **json_constraint_arg) 
        role = RhetoricalRole(**json.loads(result.message.content)).role
        results.append(role)

    return results

async def async_extract_roles(docs: List[str], llm: LLM) -> List[str]:
    json_constraint_arg = get_json_constraint_args(llm, RhetoricalRole)
    tasks = [llm.achat(craft_prompt(doc), **json_constraint_arg) 
                for doc in docs]

    results = await async_run(tasks, 15)
    results = [RhetoricalRole(**json.loads(result.message.content)).role for result in results]

    return results