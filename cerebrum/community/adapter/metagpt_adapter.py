# Modify BaseLLM method in metagpt, create fake configuration file
# Adapte metagpt to run LLM in aios
import os
from typing import Union, Optional

from cerebrum.community.adapter.adapter import FrameworkType
from .adapter import add_framework_adapter, get_request_func
from ...llm.apis import LLMQuery

try:
    from metagpt.provider.base_llm import BaseLLM
    from metagpt.const import USE_CONFIG_TIMEOUT, CONFIG_ROOT
    from metagpt.logs import logger as metagpt_logger

except ImportError:
    raise ImportError(
        "Could not import metagpt python package. "
        "Please install it with `pip install --upgrade metagpt`."
    )


@add_framework_adapter(FrameworkType.MetaGPT.value)
def prepare_metagpt():
    """
    Prepare the metagpt module to run on aios.

    This function does the following:
    1. Create a fake configuration file with effects similar to `metagpt --init-config`
    2. Replace the llm used in metagpt with aios_call
    """
    # create fake configuration file
    prepare_metagpt_config()

    # check root env
    if not os.environ.get("METAGPT_PROJECT_ROOT"):
        raise ValueError("Environment variable `METAGPT_PROJECT_ROOT` must be set")

    BaseLLM.aask = adapter_aask


async def adapter_aask(
        self,
        msg: Union[str, list[dict[str, str]]],
        system_msgs: Optional[list[str]] = None,
        format_msgs: Optional[list[dict[str, str]]] = None,
        images: Optional[Union[str, list[str]]] = None,
        timeout=USE_CONFIG_TIMEOUT,
        stream=True,
) -> str:
    if system_msgs:
        message = self._system_msgs(system_msgs)
    else:
        message = [self._default_system_msg()]
    if not self.use_system_prompt:
        message = []
    if format_msgs:
        message.extend(format_msgs)
    if isinstance(msg, str):
        message.append(self._user_msg(msg, images=images))
    else:
        message.extend(msg)
    metagpt_logger.debug(message)
    rsp = await adapter_acompletion_text(message, stream=stream, timeout=self.get_timeout(timeout))
    return rsp if rsp else ""


async def adapter_acompletion_text(
        messages: list[dict], stream: bool = False, timeout: int = USE_CONFIG_TIMEOUT
) -> str:
    """Asynchronous version of completion. Return str. Support stream-print"""
    if stream:
        print("(AIOS does not support stream mode currently. "
              "The stream mode has been automatically set to False.)")

    # call aios for response
    send_request = get_request_func()
    response = send_request(
        query=LLMQuery(
            messages=messages,
            tools=None
        )
    )["response"]

    # return text
    # text = response.response_message
    # print(f"\n{text}")
    return response["response_message"]


DEFAULT_CONFIG = """# Full Example: https://github.com/geekan/MetaGPT/blob/main/config/config2.example.yaml
# Reflected Code: https://github.com/geekan/MetaGPT/blob/main/metagpt/config2.py
llm:
  api_type: "openai"  # or azure / ollama / open_llm etc. Check LLMType for more options
  model: "xxx"  # or gpt-3.5-turbo-1106 / gpt-4-1106-preview
  base_url: "xxx"  # or forward url / other llm url
  api_key: "xxx"
"""


def prepare_metagpt_config():
    """
    Prepare a fake configuration file for MetaGPT if it does not exist.
    This configuration file is used by MetaGPT to validate the configuration.
    """
    target_path = CONFIG_ROOT / "config2.yaml"

    # Create the directory if it does not exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        # If the configuration file already exists, rename it to a backup file
        backup_path = target_path.with_suffix(".bak")
        target_path.rename(backup_path)
        print(f"Existing configuration file backed up at {backup_path}")

    # Write the default configuration to the file
    target_path.write_text(DEFAULT_CONFIG, encoding="utf-8")
    print(f"Configuration file initialized at {target_path}")
