from cerebrum import config
from cerebrum.client import Cerebrum
from cerebrum.llm.layer import LLMLayer
from cerebrum.memory.layer import MemoryLayer
from cerebrum.overrides.layer import OverridesLayer
from cerebrum.storage.layer import StorageLayer
from cerebrum.tool.layer import ToolLayer
import argparse
import os
import sys
from typing import Optional, Dict, Any


def setup_client(
    llm_name: str,
    llm_backend: str,
    root_dir: str = "root",
    memory_limit: int = 500*1024*1024,
    max_workers: int = 32,
    aios_kernel_url: str = "localhost:8000"
) -> Cerebrum:
    """Initialize and configure the Cerebrum client with specified parameters."""
    client = Cerebrum(base_url=aios_kernel_url)
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
    client: Cerebrum,
    agent_name_or_path: str,
    task: str,
    timeout: int = 300
) -> Optional[Dict[str, Any]]:
    """Run an agent with the specified task and wait for results."""
    try:
        client.connect()
        print(f"🚀 Executing agent: {os.path.basename(agent_name_or_path)}")
        print(f"📋 Task: {task}")
        
        result = client.execute(agent_name_or_path, {"task": task})
        
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
    parser = argparse.ArgumentParser(description="AIOS Agent Demo")
    parser.add_argument(
        "--llm_name", 
        help="LLM model to use",
        required=True
    )
    parser.add_argument(
        "--llm_backend",
        help="Backend service to use",
        required=True,
        choices=["openai", "google", "anthropic", "huggingface", "ollama", "vllm"]
    )
    parser.add_argument(
        "--root-dir",
        default="root",
        help="Root directory for storage"
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=500*1024*1024,
        help="Memory limit in bytes"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum number of workers"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds"
    )
    parser.add_argument(
        "--agent_name_or_path",
        # default="example/academic_agent",
        required=True,
        help="Path or name of the agent to execute"
    )
    parser.add_argument(
        "--task",
        required=True,
        # nargs="?",
        # default="Help me search the AIOS paper and introduce its core idea. ",
        help="Task for the agent to execute"
    )
    parser.add_argument(
        "--aios_kernel_url",
        default = "http://35.232.56.61:8000",
        required=True
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

        result = run_agent(
            client=client,
            agent_name_or_path=args.agent_name_or_path,
            task=args.task,
            timeout=args.timeout
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
