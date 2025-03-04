# replace run_tool_calling_llm and run_text_llm in interpreter llm
# so that interpreter can run LLM in aios
import json
import sys

from cerebrum.community.adapter.adapter import FrameworkType
from .adapter import add_framework_adapter, get_request_func
from dataclasses import dataclass

from ...llm.apis import LLMQuery

try:
    from interpreter import interpreter

except ImportError as e:
    raise ImportError(
        "Could not import interpreter python package. "
        "Please install it with `pip install open-interpreter`."
    )


@add_framework_adapter(FrameworkType.OpenInterpreter.value)
def prepare_interpreter():
    """Prepare the interpreter for running LLM in aios.
    """

    try:
        # Set the completion function in the interpreter
        interpreter.llm.completions = adapter_aios_completions
        interpreter.auto_run = True

    except Exception as e:
        print("Interpreter prepare failed: " + str(e))


@dataclass
class InterpreterFunctionAdapter:
    name: str
    arguments: str


@dataclass
class InterpreterToolCallsAdapter:
    function: InterpreterFunctionAdapter

    def __init__(self, name: str, arguments: str):
        self.function = InterpreterFunctionAdapter(name, arguments)


def adapter_aios_completions(**params):
    """aios completions replace fixed_litellm_completions in interpreter
    """

    if params.get("stream", False) is True:
        # TODO: AIOS not supprt stream mode
        print("(AIOS does not support stream mode currently. "
              "The stream mode has been automatically set to False)")
        params["stream"] = False

    # Run completion
    attempts = 2
    first_error = None

    for attempt in range(attempts):
        try:
            send_request = get_request_func()
            response = send_request(
                query=LLMQuery(
                    messages=params['messages'],
                    tools=(params["tools"] if "tools" in params else None)
                )
            )["response"]

            # format similar to completion in interpreter
            comletion = {'choices':
                [
                    {
                        'delta': {}
                    }
                ]
            }
            comletion["choices"][0]["delta"]["content"] = response["response_message"]
            if response.tool_calls is not None:
                comletion["choices"][0]["delta"]["tool_calls"] = format_tool_calls_to_interpreter(response["tool_calls"])

            return [comletion]  # If the completion is successful, exit the function
        except KeyboardInterrupt:
            print("Exiting...")
            sys.exit(0)
        except Exception as e:
            if attempt == 0:
                # Store the first error
                first_error = e

    if first_error is not None:
        raise first_error


def format_tool_calls_to_interpreter(tool_calls):
    name = tool_calls[0]["name"]
    arguments = tool_calls[0]["parameters"]
    arguments = json.dumps(arguments)
    return [InterpreterToolCallsAdapter(name, arguments)]
