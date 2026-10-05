"""Read-only SQL API over the evaluation warehouse."""

from .app import create_app

__all__ = ["create_app"]
