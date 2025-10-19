# Nodes package for ComfyUI Model Preset Pilot
# This package contains all the custom node implementations

# Only import when running in ComfyUI context
try:
    from .storage_manager import *
    __all__ = []
except ImportError:
    # Running outside ComfyUI context
    __all__ = []
