from llama_index.llms.openai_like import OpenAILike
from transformers import AutoTokenizer
import os


# def get_llm(model:str, path:str) -> OpenAILike:
#     path = os.path.join(path, model)
#     tokenizer = AutoTokenizer.from_pretrained(path)
#     llm = OpenAILike(
#         model=f"/models/{model}", api_base=os.environ['OPENAI_BASE_URL'], api_key=os.environ['OPENAI_API_KEY'], tokenizer=tokenizer
#     )

#     return llm

from llama_index.llms.together import TogetherLLM
import os


def get_llm(model:str, path:str, **kwargs) -> TogetherLLM:
    llm = TogetherLLM(
        model=model, api_key=os.environ['TOGETHER_API_KEY'], is_chat_model=False, **kwargs
    )

    return llm