# Nodes package for ComfyUI Model Preset Pilot
# This package contains all the custom node implementations

# Only import when running in ComfyUI context
try:
    from .model_preset_loader_node import ModelPresetLoader
    from .preset_creator_node import PresetCreator
    from .storage_manager import *
    __all__ = ['ModelPresetLoader', 'PresetCreator']
except ImportError:
    # Running outside ComfyUI context
    __all__ = []
