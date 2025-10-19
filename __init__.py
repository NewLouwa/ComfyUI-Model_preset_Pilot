# ComfyUI Model Preset Pilot
# A custom node module for managing model presets and configurations

from .nodes.model_preset_loader_node import ModelPresetLoaderInternal
from .nodes.preset_creator_node import PresetCreator

NODE_CLASS_MAPPINGS = {
    "ModelPresetLoaderInternal": ModelPresetLoaderInternal,
    "PresetCreator": PresetCreator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelPresetLoaderInternal": "🧭 Model Preset Loader (Internal Preview)",
    "PresetCreator": "💾 Preset Creator",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
