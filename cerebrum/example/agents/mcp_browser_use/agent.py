from cerebrum.tool.mcp_tool import mcp_pool
from cerebrum.llm.apis import llm_chat_with_tool_call_output

import asyncio
# print(mcp.name)

async def main():
    await mcp_pool.start()
    
    # playwright_client = mcp_pool.get_mcp_client("playwright")
    clients = mcp_pool.get_all_mcp_clients()
    
    tool_hints = [await client.hint() for client in clients]
    
    tool_schemas = [await client.tool_schemas() for client in clients]
        
    messages = [
        {"role": "user", "content": "search for elon musk's twitter account"},
    ]
    
    breakpoint()
    
    print(tool_hints)
    
    find_tools = {}
    for client in clients:
        for tool in await client.get_available_tools():
            find_tools[tool.name] = client.call_tool(tool.name)
    
    breakpoint()
    
    response = llm_chat_with_tool_call_output(
        agent_name="computer_use_agent",
        messages=messages,
        tools=tool_schemas,
    )
    
    tool_calls = response["response"]["tool_calls"]
    
    breakpoint()
    
    for tool_call in tool_calls:
        tool_result = await find_tools[tool_call["name"]](**tool_call["parameters"])
        breakpoint()
        print(tool_result)
    
if __name__ == "__main__":
    asyncio.run(main())
