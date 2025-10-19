# ComfyUI Model Preset Pilot
# A custom node module for managing model presets and configurations

from .nodes.model_preset_manager_node import ModelPresetManager
from .nodes.preview_image_node import PreviewImage

NODE_CLASS_MAPPINGS = {
    "ModelPresetManager": ModelPresetManager,
    "PreviewImage": PreviewImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelPresetManager": "🎛️ Model Preset Manager",
    "PreviewImage": "🖼️ Preview Image",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
