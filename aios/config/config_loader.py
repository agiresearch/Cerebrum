"""AIOS kernel config loader.

Reads ``aios/config/config.yaml`` and provides typed dataclass interfaces for
each configuration section. The ContextInjector imports ``get_config()`` at
construction time to obtain the ``WriteBarrierConfig``.

Thread-safety: the config is loaded once at module import and thereafter
immutable (frozen dataclasses). Re-loading requires calling ``reload_config()``.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Typed config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WriteBarrierConfig:
    """Configuration for the memory write barrier.

    Attributes:
        enabled: Kill-switch. When False, ``shouldInvokeBarrier`` returns
            False unconditionally so operators can disable the barrier
            without code changes.
        timeout_ms: Maximum milliseconds the barrier will wait for pending
            writes to become visible before timing out.
        poll_interval_ms: Milliseconds between visibility-probe polls during
            the bounded-wait loop.
    """

    enabled: bool = True
    timeout_ms: int = 2000
    poll_interval_ms: int = 25


@dataclass(frozen=True)
class MemoryConfig:
    """Top-level memory subsystem configuration.

    Includes both the existing kernel memory settings and the new write_barrier
    sub-section.

    Attributes:
        auto_inject: Whether the kernel automatically injects shared memories
            into LLM calls.
        auto_extract: Whether the kernel automatically extracts conversation
            memories from assistant responses.
        relevance_threshold: Minimum relevance score for a memory to be
            included in injection.
        max_injected_memories: Maximum number of memories injected per call.
        max_memory_tokens: Maximum total token count for injected memories.
        write_barrier: The write barrier sub-configuration.
    """

    auto_inject: bool = True
    auto_extract: bool = True
    relevance_threshold: float = 0.3
    max_injected_memories: int = 10
    max_memory_tokens: int = 2000
    write_barrier: WriteBarrierConfig = WriteBarrierConfig()


@dataclass(frozen=True)
class KernelConfig:
    """Root kernel configuration container.

    Attributes:
        memory: The memory subsystem configuration.
    """

    memory: MemoryConfig = MemoryConfig()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

# Default path to the config file, resolved relative to this module.
_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Module-level singleton
_config: KernelConfig | None = None


def _parse_write_barrier(raw: dict[str, Any]) -> WriteBarrierConfig:
    """Parse the write_barrier sub-section from raw YAML dict."""
    if not raw:
        return WriteBarrierConfig()
    return WriteBarrierConfig(
        enabled=bool(raw.get("enabled", True)),
        timeout_ms=int(raw.get("timeout_ms", 2000)),
        poll_interval_ms=int(raw.get("poll_interval_ms", 25)),
    )


def _parse_memory(raw: dict[str, Any]) -> MemoryConfig:
    """Parse the memory section from raw YAML dict."""
    if not raw:
        return MemoryConfig()
    wb_raw = raw.get("write_barrier", {})
    return MemoryConfig(
        auto_inject=bool(raw.get("auto_inject", True)),
        auto_extract=bool(raw.get("auto_extract", True)),
        relevance_threshold=float(raw.get("relevance_threshold", 0.3)),
        max_injected_memories=int(raw.get("max_injected_memories", 10)),
        max_memory_tokens=int(raw.get("max_memory_tokens", 2000)),
        write_barrier=_parse_write_barrier(wb_raw),
    )


def load_config(config_path: str | Path | None = None) -> KernelConfig:
    """Load and parse the kernel configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file. Defaults to the
            ``config.yaml`` shipped alongside this module. Can be overridden
            via the ``AIOS_CONFIG_PATH`` environment variable.

    Returns:
        A frozen ``KernelConfig`` instance.

    Raises:
        FileNotFoundError: If the resolved config path does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    if config_path is None:
        config_path = os.environ.get("AIOS_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH))
    path = Path(config_path)

    if not path.exists():
        # Return defaults if no config file is present
        return KernelConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    memory_raw = raw.get("memory", {})
    return KernelConfig(memory=_parse_memory(memory_raw))


def get_config() -> KernelConfig:
    """Return the module-level singleton KernelConfig, loading it if needed.

    This is the primary entry point for kernel components like the
    ContextInjector to obtain configuration at construction time.

    Returns:
        The cached ``KernelConfig`` instance.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: str | Path | None = None) -> KernelConfig:
    """Force-reload the configuration from disk.

    Useful in tests or after a config file update. Replaces the module-level
    singleton.

    Args:
        config_path: Optional override path. See ``load_config``.

    Returns:
        The freshly loaded ``KernelConfig`` instance.
    """
    global _config
    _config = load_config(config_path)
    return _config
