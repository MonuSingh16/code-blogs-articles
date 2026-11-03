"""Shared objects loaded at startup (model bundle, LLM client).

Filled by `lifespan` in main.py, read by the dependencies in deps.py.
"""

state: dict = {}
