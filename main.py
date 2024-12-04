from cerebrum import config
from cerebrum.client import Cerebrum

client = Cerebrum()
config.global_client = client

from cerebrum.llm.layer import LLMLayer
from cerebrum.memory.layer import MemoryLayer
from cerebrum.overrides.layer import OverridesLayer
from cerebrum.storage.layer import StorageLayer
from cerebrum.tool.layer import ToolLayer

client.add_llm_layer(
    LLMLayer(llm_name="gpt-4o-mini", llm_backend="openai")  # Configure your LLM
).add_storage_layer(
    StorageLayer(root_dir="root")  # Set storage directory
).add_memory_layer(
    MemoryLayer(memory_limit=104857600)  # Set memory per agent
).add_tool_layer(
    ToolLayer()  # Add tool capabilities
).override_scheduler(
    OverridesLayer(max_workers=32)  # Configure scheduling
)

try:
    # Connect to the client
    client.connect()
    # Execute your agent
    agent_path = "demo_author/demo_agent"  # Your agent's name or path
    task = "Tell me what is core idea of AIOS"       # Your task description
    result = client.execute(agent_path, {"task": task})
    
    # Get the results
    final_result = client.poll_agent(
        result["execution_id"],
        timeout=300
    )
    print("📋 Task result:", final_result)
    print("✅ Task completed")

except TimeoutError:
    print("❌ Task timed out")
except Exception as e:
    print(f"❌ Failed to execute task: {str(e)}")
finally:
    client.cleanup()