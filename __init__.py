# ComfyUI Model Preset Pilot
# A custom node module for managing model presets and configurations

# Import nodes only when running in ComfyUI context
try:
    from .nodes.model_preset_manager_node import ModelPresetManager
    from .nodes.preview_image_node import PreviewImage
    from .nodes.ksampler_helper_node import KSamplerHelper

    NODE_CLASS_MAPPINGS = {
        "ModelPresetManager": ModelPresetManager,
        "PreviewImage": PreviewImage,
        "KSamplerHelper": KSamplerHelper,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "ModelPresetManager": "🎛️ Model Preset Manager",
        "PreviewImage": "🖼️ Preview Image",
        "KSamplerHelper": "🔧 KSampler Helper",
    }
except ImportError:
    # Running outside ComfyUI context
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
