from cerebrum import config
from cerebrum.client import Cerebrum
from cerebrum.llm.layer import LLMLayer
from cerebrum.memory.layer import MemoryLayer
from cerebrum.overrides.layer import OverridesLayer
from cerebrum.storage.layer import StorageLayer
from cerebrum.tool.layer import ToolLayer
from cerebrum.manager.agent import AgentManager
import argparse
import os
import sys
from typing import Optional, Dict, Any, List
from pathlib import Path


def setup_client(
    llm_name: str,
    llm_backend: str,
    root_dir: str = "root",
    memory_limit: int = 500*1024*1024,
    max_workers: int = 32,
) -> Cerebrum:
    """Initialize and configure the Cerebrum client with specified parameters.
    
    Args:
        llm_name: Name of the LLM model to use
        root_dir: Root directory for storage
        memory_limit: Memory limit in bytes
        max_workers: Maximum number of concurrent workers
        use_backend: Backend service to use (e.g. "openai")
        
    Returns:
        Configured Cerebrum client instance
    """
    client = Cerebrum()
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


def run_tasks(
    client: Cerebrum,
    tasks: List[List[Any]],
    timeout: int = 300
) -> List[Optional[Dict[str, Any]]]:
    """Run multiple tasks and collect their results.
    
    Args:
        client: Configured Cerebrum client
        tasks: List of [agent_path, task_params] pairs
        timeout: Maximum time to wait for each task in seconds
        
    Returns:
        List of results from each task execution
    """
    try:
        client.connect()
        results = []
        final_results = []

        # Launch all tasks
        for agent_path, task_params in tasks:
            print(f"🚀 Executing agent: {os.path.basename(agent_path)}")
            print(f"📋 Task params: {task_params}")
            result = client.execute(agent_path, task_params)
            results.append(result)

        # Collect results
        for result in results:
            try:
                final_result = client.poll_agent(
                    result["execution_id"],
                    timeout=timeout
                )
                print(f"✅ Task {result['execution_id']} completed")
                final_results.append(final_result)
            except TimeoutError:
                print(f"⚠️ Task {result['execution_id']} timed out")
                final_results.append(None)
                
        return final_results

    except Exception as e:
        print(f"❌ Error during task execution: {str(e)}")
        return [None] * len(tasks)
    finally:
        client.cleanup()


def load_agent(base_url: str, agent_path: str, local: bool = True) -> Any:
    """Load an agent from local path or remote URL.
    
    Args:
        base_url: Base URL for remote agent loading
        agent_path: Path to the agent
        local: Whether to load agent locally or remotely
        
    Returns:
        Loaded agent instance
    """
    manager = AgentManager(base_url=base_url)
    agent = manager.load_agent(local=local, path=agent_path)
    return agent


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Cerebrum Task Runner")
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
        "--base-url",
        default="https://app.aios.foundation",
        help="Base URL for agent loading"
    )
    
    args = parser.parse_args()

    print("🔧 Starting Cerebrum Task Runner with configuration:")
    print(f"  LLM: {args.llm_name} ({args.llm_backend})")
    print(f"  Root Directory: {args.root_dir}")
    print(f"  Memory Limit: {args.memory_limit} bytes")
    print(f"  Max Workers: {args.max_workers}")
    print(f"  Timeout: {args.timeout} seconds")
    print(f"  Base URL: {args.base_url}")
    print("-" * 50)

    # Example tasks
    tasks = [
        ["example/academic_agent", {"task": "Help me search the AIOS paper and introduce its core idea."}],
        ["example/academic_agent", {"task": "Propose new research questions based on the AIOS paper."}]
    ]

    try:
        # Initialize client
        client = setup_client(
            llm_name=args.llm_name,
            root_dir=args.root_dir,
            memory_limit=args.memory_limit,
            max_workers=args.max_workers,
            llm_backend=args.llm_backend
        )

        # Run tasks
        results = run_tasks(
            client=client,
            tasks=tasks,
            timeout=args.timeout
        )

        # Print results
        print("\n📊 Final Results:")
        for i, result in enumerate(results, 1):
            print(f"\nTask {i} result:")
            print(result)

        return 0 if all(result is not None for result in results) else 1

    except KeyboardInterrupt:
        print("\n⚠️ Execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Execution failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())