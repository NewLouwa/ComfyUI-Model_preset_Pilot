# ComfyUI Model Preset Pilot
# A custom node module for managing model presets and configurations

from .nodes.model_preset_pilot_node import ModelPresetPilot

NODE_CLASS_MAPPINGS = {
    "ModelPresetPilot": ModelPresetPilot,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelPresetPilot": "🧭 Model Preset Pilot",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
