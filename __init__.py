# ComfyUI Model Preset Pilot
# A custom node module for managing model presets and configurations

from .nodes.model_preset_loader_node import ModelPresetLoader
from .nodes.preset_creator_node import PresetCreator
from .nodes.preview_image_node import PreviewImage, LoadLocalImage

NODE_CLASS_MAPPINGS = {
    "ModelPresetLoader": ModelPresetLoader,
    "PresetCreator": PresetCreator,
    "PreviewImage": PreviewImage,
    "LoadLocalImage": LoadLocalImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelPresetLoader": "📂 Model Preset Loader",
    "PresetCreator": "💾 Preset Creator",
    "PreviewImage": "🖼️ Preview Image",
    "LoadLocalImage": "📂 Load Local Image",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
