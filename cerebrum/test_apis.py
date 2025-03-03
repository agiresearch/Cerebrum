from cerebrum.llm.apis import llm_chat, llm_call_tool, llm_operate_file
# from cerebrum.memory.apis import alloc_memory, read_memory, write_memory, clear_memory
# from cerebrum.storage.apis import mount, retrieve, create_file, create_dir, write_file, roll_back, share_file
from cerebrum.tool.apis import call_tool

from cerebrum.interface import AutoTool

from typing import List, Dict, Any

def test_single_llm_chat(messages):
    #messages=[{"role": "user", "content": "What is the capital of France?"}]
    response = llm_chat(
        agent_name="test", 
        messages=messages, 
        base_url="http://localhost:8000", 
        llms=[
            {"name": "gpt-4o-mini","backend":"openai"}
        ]
    )
    print(response)

def test_multi_llm_chat(messages):
    #messages=[{"role": "user", "content": "What is the capital of France?"}]
    response = llm_chat(
        agent_name="test", 
        messages=messages, 
        base_url="http://localhost:8000", 
        llms=[
            {"name": "gpt-4o-mini","backend":"openai"},
            # {"name": "qwen2.5-7b","backend":"ollama"}
        ]
    )
    print(response)
    
    messages=[{"role": "user", "content": "What is the capital of United States?"}]
    response = llm_chat(
        agent_name="test", 
        messages=messages, 
        base_url="http://localhost:8000", 
        llms=[
            {"name": "gpt-4o-mini","backend":"openai"},
            # {"name": "qwen2.5:7b","backend":"ollama"}
        ]
    )
    
    print(response)
    
def test_llm_call_tool(messages):
    #messages=[{"role": "user", "content": "Tell me the core idea of OpenAGI paper"}]
    # tool = AutoTool.from_preloaded("demo_author/arxiv")
    tools = [
        {
            'type': 'function', 
            'function': {
                'name': 'demo_author/arxiv', 
                'description': 'Query articles or topics in arxiv', 
                'parameters': {
                    'type': 'object', 
                    'properties': {
                        'query': {
                            'type': 'string', 
                            'description': 'Input query that describes what to search in arxiv'
                        }
                    }, 
                    'required': ['query']
                }
            }
        }
    ]
    # breakpoint()
    # [bug remind] message should be the input while call_tool doesn't have this attribute.
    response = call_tool(agent_name="test", tool_name=tools, base_url="http://localhost:8000")
    print(response)
    
def test_operate_file():
    return NotImplementedError

def test_sto_retrieve(task):
    query_text, n, keywords = task["query_text"], task["n"], task["keywords"]
    #query_text = "top 3 papers related to KV cache"
    #n = 3
    # keywords = ["KV cache", "cache"]
    #keywords = None
    response = retrieve(agent_name="test", query_text=query_text, n=n, keywords=keywords, base_url="http://localhost:8000")
    print(response)

def test_create_file():
    return NotImplementedError

def test_create_dir():
    return NotImplementedError
        
if __name__ == "__main__":
    # agent = TestAgent("test_agent", "What is the capital of France?")
    # agent.run()
    test_single_llm_chat([{"role": "user", "content": "What is the capital of France?"}])
    # test_multi_llm_chat()
    # test_call_tool()
    # test_operate_file()
    # test_mount()
    # test_create_file()
    # test_create_dir()
