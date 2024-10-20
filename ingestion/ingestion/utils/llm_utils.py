from llama_index.llms.together import TogetherLLM
import os


def get_llm() -> TogetherLLM:
    llm = TogetherLLM(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=os.environ['TOGETHER_API_KEY']
    )

    return llm