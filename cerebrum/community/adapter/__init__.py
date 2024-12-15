from cerebrum.utils.packages import (
    is_autogen_available,
    is_metagpt_available,
    is_open_interpreter_available
)

if is_autogen_available():
    from .autogen_adapter import prepare_autogen_0_2
if is_open_interpreter_available():
    from .interpreter_adapter import prepare_interpreter
if is_metagpt_available():
    from .metagpt_adapter import prepare_metagpt
from .adapter import prepare_framework, FrameworkType, set_request_func, get_request_func

__all__ = [
    'prepare_framework',
    'FrameworkType',
    'set_request_func',
    'get_request_func'
]
