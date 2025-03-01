# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : agents_tools.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/02/20 20:36
@Description: 
"""

from huggingface_hub import login, InferenceClient
from transformers import CodeAgent, HfApiEngine

# login("")

client = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct")

def llm_engine(messages, stop_sequences=["Task"]) -> str:
    response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1000)
    answer = response.choices[0].message.content
    return answer

llm_engine = HfApiEngine(model="meta-llama/Meta-Llama-3-70B-Instruct")
agent = CodeAgent(tools=[], llm_engine=llm_engine())
