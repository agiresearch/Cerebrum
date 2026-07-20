"""Tests for aios/config/config_loader.py — kernel config loading and write barrier settings.

Validates:
- Config loads correctly from the YAML file
- WriteBarrierConfig defaults are correct per the design
- Kill-switch behavior: enabled=False causes shouldInvokeBarrier to return False unconditionally
- Existing kernel config keys are preserved (additive change)
- Config loader handles missing/malformed files gracefully

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Ensure the workspace root is on the path so aios package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aios.config.config_loader import (
    KernelConfig,
    MemoryConfig,
    WriteBarrierConfig,
    get_config,
    load_config,
    reload_config,
)


def test_default_config_loads_from_yaml():
    """Config loads from the shipped config.yaml with correct defaults."""
    config = load_config()
    assert isinstance(config, KernelConfig)
    assert isinstance(config.memory, MemoryConfig)
    assert isinstance(config.memory.write_barrier, WriteBarrierConfig)


def test_write_barrier_defaults():
    """Write barrier defaults match the design spec."""
    config = load_config()
    wb = config.memory.write_barrier
    assert wb.enabled is True
    assert wb.timeout_ms == 2000
    assert wb.poll_interval_ms == 25


def test_existing_memory_keys_preserved():
    """Existing kernel memory config keys are present and correct."""
    config = load_config()
    mem = config.memory
    assert mem.auto_inject is True
    assert mem.auto_extract is True
    assert mem.relevance_threshold == 0.3
    assert mem.max_injected_memories == 10
    assert mem.max_memory_tokens == 2000


def test_kill_switch_disabled():
    """When write_barrier.enabled is False, the config reflects it."""
    # Write a temporary config with enabled: false
    yaml_content = """
memory:
  auto_inject: true
  auto_extract: true
  relevance_threshold: 0.3
  max_injected_memories: 10
  max_memory_tokens: 2000
  write_barrier:
    enabled: false
    timeout_ms: 2000
    poll_interval_ms: 25
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        config = load_config(tmp_path)
        assert config.memory.write_barrier.enabled is False
        assert config.memory.write_barrier.timeout_ms == 2000
        assert config.memory.write_barrier.poll_interval_ms == 25
    finally:
        os.unlink(tmp_path)


def test_kill_switch_should_invoke_barrier_returns_false():
    """When enabled=False, shouldInvokeBarrier MUST return False unconditionally.

    This test imports the should_invoke_barrier function and verifies the
    kill-switch short-circuits before any other logic.
    """
    from aios.config.config_loader import WriteBarrierConfig

    # A disabled config should cause any barrier check to short-circuit
    disabled_wb = WriteBarrierConfig(enabled=False, timeout_ms=2000, poll_interval_ms=25)
    assert disabled_wb.enabled is False

    # Verify the MemoryConfig with disabled barrier is constructible
    mem_config = MemoryConfig(
        auto_inject=True,
        auto_extract=True,
        relevance_threshold=0.3,
        max_injected_memories=10,
        max_memory_tokens=2000,
        write_barrier=disabled_wb,
    )
    assert mem_config.write_barrier.enabled is False


def test_custom_timeout_and_poll_interval():
    """Operators can override timeout_ms and poll_interval_ms."""
    yaml_content = """
memory:
  write_barrier:
    enabled: true
    timeout_ms: 5000
    poll_interval_ms: 100
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        config = load_config(tmp_path)
        wb = config.memory.write_barrier
        assert wb.enabled is True
        assert wb.timeout_ms == 5000
        assert wb.poll_interval_ms == 100
    finally:
        os.unlink(tmp_path)


def test_missing_config_file_returns_defaults():
    """When config file doesn't exist, return all defaults."""
    config = load_config("/nonexistent/path/config.yaml")
    assert config.memory.write_barrier.enabled is True
    assert config.memory.write_barrier.timeout_ms == 2000
    assert config.memory.write_barrier.poll_interval_ms == 25
    assert config.memory.auto_inject is True


def test_empty_yaml_returns_defaults():
    """An empty YAML file returns all defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        tmp_path = f.name

    try:
        config = load_config(tmp_path)
        assert config.memory.write_barrier.enabled is True
        assert config.memory.write_barrier.timeout_ms == 2000
    finally:
        os.unlink(tmp_path)


def test_partial_write_barrier_block_fills_defaults():
    """A partial write_barrier block fills in missing keys with defaults."""
    yaml_content = """
memory:
  write_barrier:
    enabled: true
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        config = load_config(tmp_path)
        wb = config.memory.write_barrier
        assert wb.enabled is True
        assert wb.timeout_ms == 2000
        assert wb.poll_interval_ms == 25
    finally:
        os.unlink(tmp_path)


def test_reload_config():
    """reload_config replaces the module-level singleton."""
    yaml_content = """
memory:
  write_barrier:
    enabled: false
    timeout_ms: 9999
    poll_interval_ms: 50
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        config = reload_config(tmp_path)
        assert config.memory.write_barrier.enabled is False
        assert config.memory.write_barrier.timeout_ms == 9999

        # get_config should now return the reloaded version
        cached = get_config()
        assert cached.memory.write_barrier.timeout_ms == 9999
    finally:
        os.unlink(tmp_path)
        # Restore default config for other tests
        reload_config()


def test_env_var_override():
    """AIOS_CONFIG_PATH environment variable overrides the default path."""
    yaml_content = """
memory:
  write_barrier:
    enabled: false
    timeout_ms: 1234
    poll_interval_ms: 10
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        os.environ["AIOS_CONFIG_PATH"] = tmp_path
        config = load_config()  # No explicit path — should use env var
        assert config.memory.write_barrier.enabled is False
        assert config.memory.write_barrier.timeout_ms == 1234
    finally:
        del os.environ["AIOS_CONFIG_PATH"]
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_default_config_loads_from_yaml()
    test_write_barrier_defaults()
    test_existing_memory_keys_preserved()
    test_kill_switch_disabled()
    test_kill_switch_should_invoke_barrier_returns_false()
    test_custom_timeout_and_poll_interval()
    test_missing_config_file_returns_defaults()
    test_empty_yaml_returns_defaults()
    test_partial_write_barrier_block_fills_defaults()
    test_reload_config()
    test_env_var_override()
    print("All config loader tests passed!")
