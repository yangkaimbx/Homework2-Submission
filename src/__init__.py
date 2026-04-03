"""
Week 2: LLM Architecture & Training Lifecycle - Shared Modules

This package contains reusable code for all notebooks.

Modules with heavy dependencies (numpy, torch, transformers, etc.)
are imported on-demand in each notebook rather than eagerly here.
"""

__version__ = "2.0.0"

# Core modules (lightweight deps only: requests, anthropic, datetime)
from .llm_client import LLMClient
from .cost_tracker import CostTracker
from .utils import estimate_tokens, estimate_cost, format_response, save_task_output, append_to_reflection

__all__ = [
    'LLMClient',
    'CostTracker',
    'estimate_tokens',
    'estimate_cost',
    'format_response',
    'save_task_output',
    'append_to_reflection',
]

# Week 2 modules are imported directly in notebooks:
#   from src.attention_utils import ...
#   from src.tokenizer_utils import ...
#   from src.data_utils import ...
