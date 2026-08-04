"""TMDpolicy: executable PI0.5-to-SmolVLA distillation for LIBERO."""

from .config import LEROBOT_VERSION, ConfigError, load_config

__version__ = "0.2.0"

__all__ = ["ConfigError", "LEROBOT_VERSION", "load_config"]
