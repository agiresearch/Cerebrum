from cerebrum import config
from cerebrum.client import Cerebrum
from cerebrum.llm.layer import LLMLayer
from cerebrum.memory.layer import MemoryLayer
from cerebrum.overrides.layer import OverridesLayer
from cerebrum.storage.layer import StorageLayer
from cerebrum.tool.layer import ToolLayer
from cerebrum.manager.tool import ToolManager
import argparse
import os
import sys
from typing import Optional, Dict, Any
import json
from cerebrum.config.config_manager import config


def setup_client(
    llm_name: str,
    llm_backend: str,
    root_dir: str = None,
    memory_limit: int = None,
    max_workers: int = None,
    aios_kernel_url: str = None
) -> Cerebrum:
    """Initialize and configure the Cerebrum client with specified parameters."""
    # Use config values or override with provided parameters
    base_url = aios_kernel_url or config.get('kernel', 'base_url')
    root_dir = root_dir or config.get('client', 'root_dir')
    memory_limit = memory_limit or config.get('client', 'memory_limit')
    max_workers = max_workers or config.get('client', 'max_workers')
    
    client = Cerebrum(base_url=base_url)
    config.global_client = client

    try:
        client.add_llm_layer(LLMLayer(llm_name=llm_name, llm_backend=llm_backend)) \
              .add_storage_layer(StorageLayer(root_dir=root_dir)) \
              .add_memory_layer(MemoryLayer(memory_limit=memory_limit)) \
              .add_tool_layer(ToolLayer()) \
              .override_scheduler(OverridesLayer(max_workers=max_workers))
        
        status = client.get_status()
        print("✅ Client initialized successfully")
        print("Status:", status)
        
        return client
    except Exception as e:
        print(f"❌ Failed to initialize client: {str(e)}")
        raise


def run_agent(
    client,
    agent_name_or_path: str,
    task: str,
    timeout: int = 300,
    local_agent: bool = False
) -> Optional[Dict[str, Any]]:
    """Run an agent with the specified task and wait for results."""
    try:
        client.connect()
        print(f"🚀 Executing agent: {os.path.basename(agent_name_or_path)}")
        print(f"📋 Task: {task}")
        
        # Handle local agent path
        if local_agent:
            abs_agent_path = os.path.abspath(agent_name_or_path)
            if not os.path.exists(abs_agent_path):
                raise ValueError(f"Local agent path not found: {abs_agent_path}")
            print(f"Using local agent from: {abs_agent_path}")
            result = client.execute(
                abs_agent_path,
                {"task": task, "local_agent": True}
            )
        else:
            result = client.execute(
                agent_name_or_path,
                {"task": task}
            )
        
        try:
            final_result = client.poll_agent(
                result["execution_id"],
                timeout=timeout
            )
            print("✅ Agent execution completed")
            return final_result
        except TimeoutError:
            print("⚠️ Agent execution timed out")
            return None
            
    except Exception as e:
        print(f"❌ Error during agent execution: {str(e)}")
        return None
    finally:
        client.cleanup()


def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(description="Run an AIOS agent")
    
    # Original parameters
    parser.add_argument(
        "--llm_name",
        type=str,
        required=True,
        help="Name of the LLM to use"
    )
    
    parser.add_argument(
        "--llm_backend",
        type=str,
        required=True,
        choices=["openai", "google", "anthropic", "huggingface", "ollama", "vllm"],
        help="Backend service for the LLM"
    )
    
    parser.add_argument(
       "--agent_name_or_path",
        type=str,
        required=True,
        help="Name or path of the agent to run"
    )

    # Add new parameters
    parser.add_argument(
        "--local_agent",
        type=str,
        choices=['yes', 'no', 'true', 'false'],
        default='no',
        help="Whether to use a local agent (yes/no or true/false)"
    )

    # Add aios_kernel_url parameter
    parser.add_argument(
        "--aios_kernel_url",
        type=str,
        default="http://35.232.56.61:8000",
        required=True,
        help="URL of the AIOS kernel"
    )
    
    # Add other necessary parameters
    parser.add_argument(
        "--root_dir",
        type=str,
        default="root",
        help="Root directory for storage"
    )
    
    parser.add_argument(
        "--memory_limit",
        type=int,
        default=config.get('client', 'memory_limit'),
        help="Memory limit in bytes"
    )
    
    parser.add_argument(
        "--max_workers",
        type=int,
        default=config.get('client', 'max_workers'),
        help="Maximum number of workers"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds"
    )

    # Add a task parameter
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task for the agent to complete"
    )

    args = parser.parse_args()

    print("🔧 Starting AIOS Demo with configuration:")
    print(f"  LLM: {args.llm_name} ({args.llm_backend})")
    print(f"  Root Directory: {args.root_dir}")
    print(f"  Memory Limit: {args.memory_limit} bytes")
    print(f"  Max Workers: {args.max_workers}")
    print(f"  Timeout: {args.timeout} seconds")
    print(f"  Agent: {args.agent_name_or_path}")
    print(f"  Task: {args.task}")
    print("-" * 50)

    try:
        client = setup_client(
            llm_name=args.llm_name,
            llm_backend=args.llm_backend,
            root_dir=args.root_dir,
            memory_limit=args.memory_limit,
            max_workers=args.max_workers,
            aios_kernel_url=args.aios_kernel_url
        )

        # Convert local_agent string to boolean value
        is_local_agent = args.local_agent.lower() in ['yes', 'true']

        result = run_agent(
            client=client,
            agent_name_or_path=args.agent_name_or_path,
            task=args.task,
            timeout=args.timeout,
            local_agent=is_local_agent
        )

        if result:
            print("\n📊 Final Result:")
            print(result)
            return 0
        return 1

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
