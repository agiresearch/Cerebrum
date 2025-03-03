# Support third-party frameworks running on AIOS.
import functools
from enum import Enum
from typing import Callable

from cerebrum.llm.apis import llm_chat


class FrameworkType(Enum):
    """
    Enum for the types of frameworks that AIOS supports.

    This Enum contains the types of frameworks that AIOS supports.
    """
    MetaGPT = "MetaGPT"
    OpenInterpreter = "Open-Interpreter"
    AutoGen = "AutoGen~0.2"


FRAMEWORK_ADAPTER = {}

REQUEST_FUNC = None

def add_framework_adapter(framework_type: str):
    """
    Decorator to register a framework adapter function.

    This function takes a framework type as an argument and returns a decorator.
    The returned decorator can be used to wrap a function, which will then be
    registered as the adapter for the specified framework type in the global
    FRAMEWORK_ADAPTER dictionary.

    Args:
        framework_type (str): The type of framework for which the adapter function
        is being registered.
    """
    def add_framework_adapter_helper(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        FRAMEWORK_ADAPTER[framework_type] = func
        return wrapper

    return add_framework_adapter_helper


def prepare_framework(framework_type: FrameworkType):
    """
    Prepare the framework adapter for AIOS.

    This function is called by the framework before the user code is executed.
    It will call the adapter function associated with the given framework type.
    If the framework is not supported, it will log a warning message.

    Args:
        framework_type (str): The type of framework to prepare.
    """
    if framework_type.value not in FRAMEWORK_ADAPTER:
        print(f"[ERROR] Framework {framework_type.value} are not supported")
    else:
        print(f"Prepare framework {framework_type.value} sucessfully.")
        adapter = FRAMEWORK_ADAPTER[framework_type.value]
        adapter()


def set_request_func(request_func: Callable, agent_name: str, base_url: str = "http://localhost:8000"):
    """
    Set the request function for the AIOS adapter.

    Args:
        request_func (Callable): The function to handle requests.
        agent_name (str): The name of the agent.
        base_url (str): The base URL of the AIOS.

    Returns:
        None
    """
    def request_wrapper(query):
        return request_func(agent_name=agent_name, query=query, base_url=base_url)

    global REQUEST_FUNC
    REQUEST_FUNC = request_wrapper


def get_request_func():
    """
    Get the request function set by the AIOS adapter.

    Returns:
        Callable: The function to handle requests.
    """
    return REQUEST_FUNC
