# ComfyUI Model Preset Pilot
# A custom node module for managing model presets and configurations

# Import nodes only when running in ComfyUI context
try:
    from .nodes.model_preset_manager_node import ModelPresetManager
    from .nodes.model_preset_loader_node import ModelPresetLoader
    from .nodes.preset_creator_node import PresetCreator
    from .nodes.preview_image_node import PreviewImage

    NODE_CLASS_MAPPINGS = {
        "ModelPresetManager": ModelPresetManager,
        "ModelPresetLoader": ModelPresetLoader,
        "PresetCreator": PresetCreator,
        "PreviewImage": PreviewImage,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelPresetManager": "🎛️ Model Preset Manager",
        "ModelPresetLoader": "📂 Model Preset Loader",
        "PresetCreator": "💾 Preset Creator",
        "PreviewImage": "🖼️ Preview Image",
    }
except ImportError:
    # Running outside ComfyUI context
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
