# Cerebrum: Agent SDK for AIOS

<a href='https://docs.aios.foundation/'><img src='https://img.shields.io/badge/Documentation-Cerebrum-green'></a>
[![Code License](https://img.shields.io/badge/Code%20License-MIT-orange.svg)](https://github.com/agiresearch/AIOS/blob/main/LICENSE)
<a href='https://discord.gg/B2HFxEgTJX'><img src='https://img.shields.io/badge/Community-Discord-8A2BE2'></a>

The goal of AIOS is to build a Large Language Model (LLM) agent operating system, which intends to embed large language model into the operating system as the brain of the OS. AIOS is designed to address problems (e.g., scheduling, context switch, memory management, etc.) during the development and deployment of LLM-based agents, for a better ecosystem among agent developers and users.

## 🏠 Cerebrum Architecture
<p align="center">
<img src="docs/assets/details.png">
</p>

The AIOS-Agent SDK is designed for agent users and developers, enabling them to build and run agent applications by interacting with the [AIOS kernel](https://github.com/agiresearch/AIOS.git). 

## 📰 News
- **[2024-11-26]** 🔥 Cerebrum is available for public release on PyPI!

## Installation

### Install From Source
1. **Clone Repo**
   ```bash
   git clone https://github.com/agiresearch/Cerebrum.git

   cd Cerebrum
   ```

3. **Create Virtual Environment**
   ```bash
   conda create -n cerebrum-env python=3.10
   ```
   or
   ```bash
   conda create -n cerebrum-env python=3.11
   ```
   or
   ```bash
   # Windows (cmd)
   python -m venv cerebrum-env

   # Linux/MacOS
   python3 -m venv cerebrum-env
   ```

4. **Activate the environment**
   ```bash
   conda activate myenv
   ```
   or
   ```bash
   # Windows (cmd)
   cd cerebrum-env
   cd Scripts
   activate.bat
   cd ..
   cd ..
   

   # Linux/MacOS
   source cerebrum-env/bin/activate
   ```

6. **Install the package**
   ```bash
   pip install -e .
   ```

7. **Verify installation**
   ```bash
   python -c "import cerebrum; from cerebrum.client import Cerebrum; print(Cerebrum)"
   ```

## ✈️ Quickstart
> [!TIP] 
>
> Please see our [documentation](https://docs.aios.foundation/) for more information.

1. **Start the AIOS Kernel** 
   📝 See [here](https://docs.aios.foundation/getting-started/installation).

2. **Run the AIOS Client**

   Run an agent using the client
   ```bash
   run-agent --llm_name gpt-4o-mini --llm_backend openai --agent_name_or_path <agent name or agent path> --task <task that agent needs to complete>
   ```
   For the agent_name_or_path argument, you can either pass the agent names (in the format of author_name/agent_name) that are available on the [agenthub](https://app.aios.foundation/agenthub), or pass the absolute path for the agent folder in your computer. 

   For example, you can run a demo agent avaiable on the agenthub using the following command:  
   ```bash
   run-agent --llm_name gpt-4o-mini --llm_backend openai --agent_name_or_path demo_author/demo_agent --task "Tell me what is core idea of AIOS" --aios_kernel_url http://localhost:8000
   ```
   or you can run the demo agent by passing the absolute path of the agent folder in your computer
   ```bash
   run-agent --llm_name gpt-4o-mini --llm_backend openai --agent_name_or_path <your_local_folder_of_cerebrum>/cerebrum/example/agents/demo_agent --task "Tell me what is core idea of AIOS" --aios_kernel_url "http://localhost:8000"
   ```

   Code file is located at `cerebrum/example/run_agent.py`

## 👤 Getting Started with Your Client

Let's walk through how to set up and customize your client to work with the AIOS kernel. We'll break this down into simple steps.

### Step 1: Initialize Your Client
First, let's create your client instance:
```python
from cerebrum import config
from cerebrum.client import Cerebrum

aios_kernel_url = "http://localhost:8000"

client = Cerebrum(base_url = aios_kernel_url)
config.global_client = client
```

### Step 2: Add Functionality Layers
The AIOS kernel offers five core modules you can customize:
- LLM (Language Model)
- Memory
- Storage
- Tools
- Scheduler

Here's how to add these layers to your client:
```python
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
```

### Step 3: Run Your Agent
Now you can run agents and get their results:
```python
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
```

You can find all these agents in the [example agents folder](./cerebrum/example/agents/). If you would like to customize and develop your new agents, you can check out the guides on [Developing New Agents](#-develop-and-customize-new-agents) and [Developing New Tools](#develop-and-customize-new-tools).

## 🚀 Develop and customize new agents

This guide will walk you through creating and publishing your own agents for AIOS. 
### Agent Structure

First, let's look at how to organize your agent's files. Every agent needs three essential components:

```
author/
└── agent_name/
      │── entry.py        # Your agent's main logic
      │── config.json     # Configuration and metadata
      └── meta_requirements.txt  # Additional dependencies
```

For example, if your name is 'example' and you're building a demo_agent that searches and summarizes articles, your folder structure would look like this:

```
example/
   └── demo_agent/
         │── entry.py
         │── config.json
         └── meta_requirements.txt
```

Note: If your agent needs any libraries beyond AIOS's built-in ones, make sure to list them in meta_requirements.txt. Apart from the above three files, you can have any other files in your folder. 

### Configure the agent

#### Set up Metadata

Your agent needs a config.json file that describes its functionality. Here's what it should include:

```json
{
   "name": "demo_agent",
   "description": [
      "Demo agent that can help search AIOS-related papers"
   ],
   "tools": [
      "demo_author/arxiv"
   ],
   "meta": {
      "author": "demo_author",
      "version": "0.0.1",
      "license": "CC0"
   },
   "build": {
      "entry": "agent.py",
      "module": "DemoAgent"
   }
}
```

### Available tools

When setting up your agent, you'll need to specify which tools it will use. Below is a list of all currently available tools and how to reference them in your configuration:

| Author | Name | How to call them |
|:--|:--|:--|
| example | arxiv | example/arxiv |
| example | bing_search | example/bing_search |
| example | currency_converter | example/currency_converter |
| example | wolfram_alpha | example/wolfram_alpha |
| example | google_search | example/google_search |
| openai | speech_to_text | openai/speech_to_text |
| example | web_browser | example/web_browser |
| timbrooks | image_to_image | timbrooks/image_to_image |
| example | downloader | example/downloader |
| example | doc_question_answering | example/doc_question_answering |
| stability-ai | text_to_image | stability-ai/text_to_image |
| example | text_to_speech | example/text_to_speech |

To use these tools in your agent, simply include their reference (from the "How to Use" column) in your agent's configuration file. For example, if you want your agent to be able to search academic papers and convert currencies, you would include both `example/arxiv` and `example/currency_converter` in your configuration.

If you would like to create your new tools, you can either integrate the tool within your agent code or you can follow the tool examples in the [tool folder](./cerebrum/example/tools/) to develop your standalone tools. The detailed instructions are in [How to develop new tools](#develop-and-publish-new-tools)

### Build Agent

Let's walk through creating your agent's core functionality.

#### Set up the Base Agent Class

First, create your agent class by inheriting from BaseAgent:

```python
from cerebrum.agents.base import BaseAgent
from cerebrum.llm.communication import LLMQuery
import json

class DemoAgent(BaseAgent):
    def __init__(self, agent_name, task_input, config_):
        super().__init__(agent_name, task_input, config_)

        self.plan_max_fail_times = 3
        self.tool_call_max_fail_times = 3

        self.start_time = None
        self.end_time = None
        self.request_waiting_times: list = []
        self.request_turnaround_times: list = []
        self.task_input = task_input
        self.messages = []
        self.workflow_mode = "manual"  # (manual, automatic)
        self.rounds = 0
```

#### Import Query Functions

AIOS provides several `Query` classes for different types of interactions and use the `Response` class in [here](./cerebrum/llm/communication.py) to receive results from the AIOS kernel. 

| Query Class | Arguments | Output |
|:--|:--|:--|
| `LLMQuery` | messages: `List`, tools: `List`, action_type: `str`, message_return_type: `str` | response: `Response` |
| `MemoryQuery` | TBD | response: `Response` |
| `StorageQuery` | TBD | response: `Response` |
| `ToolQuery` | tool_calls: `List` | response: `Response` |

Here's how to import a specific query
```python
from cerebrum.llm.communication import LLMQuery  # Using LLMQuery as an example
```

#### Construct system instructions

Here's how to set up your agent's system instructions and you need to put this function inside your agent class

```python
def build_system_instruction(self):
    prefix = "".join(["".join(self.config["description"])])

    plan_instruction = "".join(
        [
            f"You are given the available tools from the tool list: {json.dumps(self.tool_info)} to help you solve problems. ",
            "Generate a plan with comprehensive yet minimal steps to fulfill the task. ",
            "The plan must follow the json format as below: ",
            "[",
            '{"action_type": "action_type_value", "action": "action_value","tool_use": [tool_name1, tool_name2,...]}',
            '{"action_type": "action_type_value", "action": "action_value", "tool_use": [tool_name1, tool_name2,...]}',
            "...",
            "]",
            "In each step of the planned plan, identify tools to use and recognize no tool is necessary. ",
            "Followings are some plan examples. ",
            "[" "[",
            '{"action_type": "tool_use", "action": "gather information from arxiv. ", "tool_use": ["arxiv"]},',
            '{"action_type": "chat", "action": "write a summarization based on the gathered information. ", "tool_use": []}',
            "];",
            "[",
            '{"action_type": "tool_use", "action": "gather information from arxiv. ", "tool_use": ["arxiv"]},',
            '{"action_type": "chat", "action": "understand the current methods and propose ideas that can improve ", "tool_use": []}',
            "]",
            "]",
        ]
    )

    if self.workflow_mode == "manual":
        self.messages.append({"role": "system", "content": prefix})

    else:
        assert self.workflow_mode == "automatic"
        self.messages.append({"role": "system", "content": prefix})
        self.messages.append({"role": "user", "content": plan_instruction})
```

#### Create Workflows

You can create a workflow for the agent to execute its task and you need to put this function inside your agent class. 

Manual workflow example:
```python
def manual_workflow(self):
    workflow = [
        {
            "action_type": "tool_use",
            "action": "Search for relevant papers",
            "tool_use": ["demo_author/arxiv"],
        },
        {
            "action_type": "chat",
            "action": "Provide responses based on the user's query",
            "tool_use": [],
        },
    ]
    return workflow
```


#### Implement the Run Method

Finally, implement the run method to execute your agent's workflow and you need to put this function inside your agent class. 

```python
def run(self):
    self.build_system_instruction()

    task_input = self.task_input

    self.messages.append({"role": "user", "content": task_input})

    workflow = None

    if self.workflow_mode == "automatic":
        workflow = self.automatic_workflow()
        self.messages = self.messages[:1]  # clear long context

    else:
        assert self.workflow_mode == "manual"
        workflow = self.manual_workflow()

    self.messages.append(
        {
            "role": "user",
            "content": f"[Thinking]: The workflow generated for the problem is {json.dumps(workflow)}. Follow the workflow to solve the problem step by step. ",
        }
    )

    try:
        if workflow:
            final_result = ""

            for i, step in enumerate(workflow):
                action_type = step["action_type"]
                action = step["action"]
                tool_use = step["tool_use"]

                prompt = f"At step {i + 1}, you need to: {action}. "
                self.messages.append({"role": "user", "content": prompt})

                if tool_use:
                    selected_tools = self.pre_select_tools(tool_use)

                else:
                    selected_tools = None

                response = self.send_request(
                    agent_name=self.agent_name,
                    query=LLMQuery(
                        messages=self.messages,
                        tools=selected_tools,
                        action_type=action_type,
                    ),
                )["response"]
                
                self.messages.append({"role": "assistant", "content": response.response_message})

                self.rounds += 1


            final_result = self.messages[-1]["content"]
            
            return {
                "agent_name": self.agent_name,
                "result": final_result,
                "rounds": self.rounds,
            }

        else:
            return {
                "agent_name": self.agent_name,
                "result": "Failed to generate a valid workflow in the given times.",
                "rounds": self.rounds,
            }
            
    except Exception as e:
        return {}
```

### Run the Agent
To test your agent, use the run_agent command to run:

```bash
run-agent --llm_name <llm_name> --llm_backend <llm_backend> --agent_name_or_path <agent_name_or_path> --task <task_input> --aios_kernel_url <aios_kernel_url>
```
Replace the placeholders with your specific values:
- `<llm_name>`: The name of the language model you want to use
- `<llm_backend>`: The backend service for the language model
- `<your_agent_folder_path>`: The path to your agent's folder
- `<task_input>`: The task you want your agent to complete
- `<aios_kernel_url>`: The url that is connected to the aios kernel

or you can run the agent using the source code in the cerebrum/example/run_agent
```bash
python cerebrum/example/run_agent --llm_name <llm_name> --llm_backend <llm_backend> --agent_name_or_path <agent_name_or_path>> --task <task_input> --aios_kernel_url <aios_kernel_url>
```

## 🔧Develop and Customize New Tools
### Tool Structure
Similar as developing new agents, developing tools also need to follow a simple directory structure:
```
demo_author/
└── demo_tool/
    │── entry.py      # Contains your tool's main logic
    └── config.json   # Tool configuration and metadata
```

### Setting up config.json
Your tool needs a configuration file that describes its properties. Here's an example of how to set it up:

```json
{
    "name": "arxiv",
    "description": [
        "The arxiv tool that can be used to search for papers on arxiv"
    ],
    "meta": {
        "author": "demo_author",
        "version": "1.0.6",
        "license": "CC0"
    },
    "build": {
        "entry": "tool.py",
        "module": "Arxiv"
    }
}
```
### Create Tool Class
In `entry.py`, you'll need to implement a tool class which is identified in the config.json with two essential methods:

1. `get_tool_call_format`: Defines how LLMs should interact with your tool
2. `run`: Contains your tool's main functionality

Here's an example:

```python
class Arxiv:
    def get_tool_call_format(self):
        tool_call_format = {
            "type": "function",
            "function": {
                "name": "demo_author/arxiv",
                "description": "Query articles or topics in arxiv",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Input query that describes what to search in arxiv"
                        }
                    },
                    "required": [
                        "query"
                    ]
                }
            }
        }
        return tool_call_format

    def run(self, params: dict):
        """
        Main tool logic goes here.
        Args:
            params: Dictionary containing tool parameters
        Returns:
            Your tool's output
        """
        # Your code here
        result = do_something(params['param_name'])
        return result
```

### Integration Tips
When integrating your tool for the agents you develop:
- Use absolute paths to reference your tool in agent configurations
- Example: `/path/to/your/tools/example/your_tool` instead of just `author/tool_name`

## Supported LLM Cores
| Provider 🏢 | Model Name 🤖 | Open Source 🔓 | Model String ⌨️ | Backend ⚙️ | Required API Key |
|:------------|:-------------|:---------------|:---------------|:---------------|:----------------|
| Anthropic | Claude 3.5 Sonnet | ❌ | claude-3-5-sonnet-20241022 |anthropic | ANTHROPIC_API_KEY |
| Anthropic | Claude 3.5 Haiku | ❌ | claude-3-5-haiku-20241022 |anthropic | ANTHROPIC_API_KEY |
| Anthropic | Claude 3 Opus | ❌ | claude-3-opus-20240229 |anthropic | ANTHROPIC_API_KEY |
| Anthropic | Claude 3 Sonnet | ❌ | claude-3-sonnet-20240229 |anthropic | ANTHROPIC_API_KEY |
| Anthropic | Claude 3 Haiku | ❌ | claude-3-haiku-20240307 |anthropic | ANTHROPIC_API_KEY |
| OpenAI | GPT-4 | ❌ | gpt-4 |openai| OPENAI_API_KEY |
| OpenAI | GPT-4 Turbo | ❌ | gpt-4-turbo |openai| OPENAI_API_KEY |
| OpenAI | GPT-4o | ❌ | gpt-4o |openai| OPENAI_API_KEY |
| OpenAI | GPT-4o mini | ❌ | gpt-4o-mini |openai| OPENAI_API_KEY |
| OpenAI | GPT-3.5 Turbo | ❌ | gpt-3.5-turbo |openai| OPENAI_API_KEY |
| Google | Gemini 1.5 Flash | ❌ | gemini-1.5-flash |google| GEMINI_API_KEY |
| Google | Gemini 1.5 Flash-8B | ❌ | gemini-1.5-flash-8b |google| GEMINI_API_KEY |
| Google | Gemini 1.5 Pro | ❌ | gemini-1.5-pro |google| GEMINI_API_KEY |
| Google | Gemini 1.0 Pro | ❌ | gemini-1.0-pro |google| GEMINI_API_KEY |
| Groq | Llama 3.2 90B Vision | ✅ | llama-3.2-90b-vision-preview |groq| GROQ_API_KEY |
| Groq | Llama 3.2 11B Vision | ✅ | llama-3.2-11b-vision-preview |groq| GROQ_API_KEY |
| Groq | Llama 3.1 70B | ✅ | llama-3.1-70b-versatile |groq| GROQ_API_KEY |
| Groq | Llama Guard 3 8B | ✅ | llama-guard-3-8b |groq| GROQ_API_KEY |
| Groq | Llama 3 70B | ✅ | llama3-70b-8192 |groq| GROQ_API_KEY |
| Groq | Llama 3 8B | ✅ | llama3-8b-8192 |groq| GROQ_API_KEY |
| Groq | Mixtral 8x7B | ✅ | mixtral-8x7b-32768 |groq| GROQ_API_KEY |
| Groq | Gemma 7B | ✅ | gemma-7b-it |groq| GROQ_API_KEY |
| Groq | Gemma 2B | ✅ | gemma2-9b-it |groq| GROQ_API_KEY |
| Groq | Llama3 Groq 70B | ✅ | llama3-groq-70b-8192-tool-use-preview |groq| GROQ_API_KEY |
| Groq | Llama3 Groq 8B | ✅ | llama3-groq-8b-8192-tool-use-preview |groq| GROQ_API_KEY |
| ollama | [All Models](https://ollama.com/search) | ✅ | model-name |ollama| - |
| vLLM | [All Models](https://docs.vllm.ai/en/latest/) | ✅ | model-name |vllm| - |
| HuggingFace | [All Models](https://huggingface.co/models/) | ✅ | model-name |huggingface| HF_HOME |


## 🖋️ References
```
@article{mei2024aios,
  title={AIOS: LLM Agent Operating System},
  author={Mei, Kai and Li, Zelong and Xu, Shuyuan and Ye, Ruosong and Ge, Yingqiang and Zhang, Yongfeng}
  journal={arXiv:2403.16971},
  year={2024}
}
@article{ge2023llm,
  title={LLM as OS, Agents as Apps: Envisioning AIOS, Agents and the AIOS-Agent Ecosystem},
  author={Ge, Yingqiang and Ren, Yujie and Hua, Wenyue and Xu, Shuyuan and Tan, Juntao and Zhang, Yongfeng},
  journal={arXiv:2312.03815},
  year={2023}
}
```

## 🚀 Contributions
For how to contribute, see [CONTRIBUTE](https://github.com/agiresearch/Cerebrum/blob/main/CONTRIBUTE.md). If you would like to contribute to the codebase, [issues](https://github.com/agiresearch/Cerebrum/issues) or [pull requests](https://github.com/agiresearch/Cerebrum/pulls) are always welcome!

## 🌍 Cerebrum Contributors
[![Cerebrum contributors](https://contrib.rocks/image?repo=agiresearch/Cerebrum&max=300)](https://github.com/agiresearch/Cerebrum/graphs/contributors)


## 🤝 Discord Channel
If you would like to join the community, ask questions, chat with fellows, learn about or propose new features, and participate in future developments, join our [Discord Community](https://discord.gg/B2HFxEgTJX)!

## 📪 Contact

For issues related to Cerebrum development, we encourage submitting [issues](https://github.com/agiresearch/Cerebrum/issues), [pull requests](https://github.com/agiresearch/Cerebrum/pulls), or initiating discussions in AIOS [Discord Channel](https://discord.gg/B2HFxEgTJX). For other issues please feel free to contact the AIOS Foundation ([contact@aios.foundation](mailto:contact@aios.foundation)).




