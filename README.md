# Cerebrum: Agent SDK for AIOS

<a href='https://docs.aios.foundation/'><img src='https://img.shields.io/badge/Documentation-Cerebrum-green'></a>
[![Code License](https://img.shields.io/badge/Code%20License-MIT-orange.svg)](https://github.com/agiresearch/AIOS/blob/main/LICENSE)
<a href='https://discord.gg/B2HFxEgTJX'><img src='https://img.shields.io/badge/Community-Discord-8A2BE2'></a>

AIOS is the AI Agent Operating System, which embeds large language model (LLM) into the operating system and facilitates the development and deployment of LLM-based AI Agents. AIOS is designed to address problems (e.g., scheduling, context switch, memory management, storage management, tool management, Agent SDK management, etc.) during the development and deployment of LLM-based agents, towards a better AIOS-Agent ecosystem for agent developers and agent users. AIOS includes the AIOS Kernel (the [AIOS](https://github.com/agiresearch/AIOS) repository) and the AIOS SDK (this [Cerebrum](https://github.com/agiresearch/Cerebrum) repository). AIOS supports both Web UI and Terminal UI.


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

2. **Create Virtual Environment**
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

3. **Activate the environment**
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

4. **Install the package**  
   Using uv (Recommended)
   ```bash
   pip install uv
   uv pip install -e .
   ```
   or using pip
   ```
   pip install -e .
   ```

5. **Verify installation**
   ```bash
   python -c "import cerebrum; from cerebrum.client import Cerebrum; print(Cerebrum)"
   ```

## ✈️ Quickstart
> [!TIP] 
>
> Please see our [documentation](https://docs.aios.foundation/) for more information.

### 1. Start the AIOS Kernel
📝 See [here](https://docs.aios.foundation/getting-started/installation).

Below are some useful commands to use
- [List available LLMs](./cerebrum/commands/list_available_llms.py)
    ```bash
    list-available-llms
    ```

- [List agents from agenthub](./cerebrum/commands/list_agenthub_agents.py)
    ```bash
    list-agenthub-agents
    ```
- [List agents from local](./cerebrum/commands/list_local_agents.py)
    ```
    list-local-agents
    ```
- [Download agents](./cerebrum/commands/download_agent.py)
    ```bash
    download-agent \
        --agent_author <agent_author> \
        --agent_name <agent_name> \
        --agent_version <agent_version> \
        --agenthub_url <agenthub_url>
    ```
- [Upload agent](./cerebrum/commands/upload_agent.py)
    ```bash
    upload-agents \
        --agent_path <agent_path> \
        --agenthub_url <agenthub_url>
    ```

- [List tools from toolhub](./cerebrum/commands/list_toolhub_tools.py)
    ```bash
    list-toolhub-tools
    ```
- [List tools from local](./cerebrum/commands/list_local_tools.py)
    ```bash
    list-local-tools
    ```
- [Download tool](./cerebrum/commands/download_tool.py)
    ```bash
    download-tool \
        --tool_author <tool_author> \
        --tool_name <tool_name> \
        --tool_version <tool_version> \
        --toolhub_url <toolhub_url>
    ```
- [Upload tool](./cerebrum/commands/upload_tool.py)
    ```bash
    upload-tool \
        --tool_path <tool_path> \
        --toolhub_url <toolhub_url>
    ```

### 2. Run agents

Either run agents that already exist in the local by passing the path to the agent directory

```
run-agent \
    --mode local \
    --agent_path <agent_name_or_path> \ # path to the agent directory
    --task <task_input> \
    --agenthub_url <agenthub_url>
```

For example, to run the test_agent in the local directory, you can run:

```
run-agent \
    --mode local \
    --agent_path cerebrum/example/agents/test_agent \
    --task "What is the capital of United States?"
```

Or run agents that are uploaded to agenthub by passing the author and agent name

```
run-agent \
    --mode remote \
    --agent_author <author> \
    --agent_name <agent_name> \
    --agent_version <agent_version> \
    --task <task_input> \
    --agenthub_url <agenthub_url>
```

For example, to run the test_agent in the agenthub, you can run:

```
run-agent \
    --mode remote \
    --agent_author example \
    --agent_name test_agent \
    --agent_version 0.0.3 \
    --task "What is the capital of United States?" \
    --agenthub_url https://app.aios.foundation
```


## 🚀 Develop and customize new agents

This guide will walk you through creating and publishing your own agents for AIOS. 
### Agent Structure

First, let's look at how to organize your agent's files. Every agent needs three essential components:

```
author_name/
└── agent_name/
      │── entry.py        # Your agent's main logic
      │── config.json     # Configuration and metadata
      └── meta_requirements.txt  # Additional dependencies
```

For example, if your name is 'demo_author' and you're building a demo_agent that searches and summarizes articles, your folder structure would look like this:

```
demo_author/
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

### APIs to build your agents
- [LLM APIs](./cerebrum/llm/apis.py)
- [Memory APIs](./cerebrum/memory/apis.py)
- [Storage APIs](./cerebrum/storage/apis.py)
- [Tool APIs](./cerebrum/tool/apis.py)

### Available tools

There are two ways to use tools in your agents:

#### 1. Use tools from ToolHub

You can list all available tools in the ToolHub using the following command:

```bash
list-toolhub-tools
```

This will display all tools available in the remote ToolHub. 

To load a tool from ToolHub in your code:

```python
from cerebrum.interface import AutoTool
tool = AutoTool.from_preloaded("example/arxiv", local=False)
```

#### 2. Use tools from local folders

You can also list tools available in your local environment using the following command:

```bash
list-local-tools
```

To load a local tool in your code:

```python
from cerebrum.tool import AutoTool
tool = AutoTool.from_preloaded("google/google_search", local=True)
```

If you would like to create your new tools, refer to [How to develop new tools](#develop-and-publish-new-tools)

### How to upload your agents to the agenthub
Run the following command to upload your agents to the agenthub:

```python
python cerebrum/upload_agent.py \
    --agent_path <agent_path> \ # agent path to the agent directory
    --agenthub_url <agenthub_url> # the url of the agenthub, default is https://app.aios.foundation
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

> [!IMPORTANT]
> To use the agents in your local device, you need to put the tool folder under the cerebrum/tool/core folder and register your tool in the cerebrum/tool/core/registry.py

### Create Tool Class
In `entry.py`, you'll need to implement a tool class which is identified in the config.json with two essential methods:

1. `get_tool_call_format`: Defines how LLMs should interact with your tool
2. `run`: Contains your tool's main functionality

Here's an example:

```python
class Wikipedia:
    def __init__(self):
        super().__init__()
        self.WIKIPEDIA_MAX_QUERY_LENGTH = 300
        self.top_k_results = 3
        self.lang = "en"
        self.load_all_available_meta: bool = False
        self.doc_content_chars_max: int = 4000
        self.wiki_client = self.build_client()

    def build_client(self):
        try:
            import wikipedia
            wikipedia.set_lang(self.lang)

        except ImportError:
            raise ImportError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )
        return wikipedia

    def run(self, params) -> str:
        """Run Wikipedia search and get page summaries."""
        query = params["query"]
        page_titles = self.wiki_client.search(query, results=self.top_k_results)
        summaries = []
        for page_title in page_titles[: self.top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                if summary := self._formatted_page_summary(page_title, wiki_page):
                    summaries.append(summary)
        if not summaries:
            return "No good Wikipedia Search Result was found"
        return "\n\n".join(summaries)[: self.doc_content_chars_max]

    @staticmethod
    def _formatted_page_summary(page_title: str, wiki_page: Any) -> Optional[str]:
        return f"Page: {page_title}\nSummary: {wiki_page.summary}"

    def get_tool_call_format(self):
        tool_call_format = {
			"type": "function",
			"function": {
				"name": "wikipedia",
				"description": "Provides relevant information about the destination",
				"parameters": {
					"type": "object",
					"properties": {
						"query": {
							"type": "string",
							"description": "Search query for Wikipedia"
						}
					},
					"required": [
						"query"
					]
				}
			}
		}
        return tool_call_format
```


### How to publish tools to the toolhub
Before publishing tools, you need to set up the configurations as the following: 

```json
{
    "name": "wikipedia",
    "description": [
        "Search information in the wikipedia"
    ],
    "meta": {
        "author": "example",
        "version": "0.0.1",
        "license": "CC0"
    },
    "build": {
        "entry": "tool.py",
        "module": "Wikipedia"
    }
}
```

then you can use the following command to upload tool

```python
python cerebrum/commands/upload_tool.py \
    --tool_path <tool_path> \ # tool path to the tool directory
    --toolhub_url <toolhub_url> # the url of the toolhub, default is https://app.aios.foundation
```

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




