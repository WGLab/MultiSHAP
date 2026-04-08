"""MultiSHAP: research software for multimodal interaction analysis.

This package provides a formal Python package entry point for the MultiSHAP
repository. The current implementation focuses on packaging, metadata, and a
stable command-line interface for the existing analysis workflow.
"""

from .runner import run_legacy_cli

__all__ = ["run_legacy_cli"]
__version__ = "0.1.0"
