"""
Preset to KSampler Converter Node
Converts preset parameters to KSampler-compatible inputs
"""

import os
import torch
import numpy as np
from PIL import Image
import comfy.utils
import nodes
import folder_paths

# Get available samplers/schedulers from ComfyUI core (source of truth)
def _get_sampler_choices():
    """Get available samplers from comfy.samplers (source of truth)"""
    try:
        import comfy.samplers as comfy_samplers
        
        # Get the global registry of registered samplers
        samplers = getattr(comfy_samplers, "SAMPLERS", [])
        
        # Remove None/empty values and duplicates
        samplers = [s for s in samplers if s]
        samplers = sorted(set(samplers))
        
        # Optional: verify samplers are actually available in the lookup
        registered = getattr(comfy_samplers, "SAMPLER_LOOKUP", {})
        valid_samplers = [s for s in samplers if s in registered and callable(registered.get(s))]
        
        if valid_samplers:
            print(f"[PresetToKSampler] Found {len(valid_samplers)} valid samplers from comfy.samplers")
            return valid_samplers
        elif samplers:
            print(f"[PresetToKSampler] Found {len(samplers)} samplers from comfy.samplers (not all verified)")
            return samplers
        else:
            print("[PresetToKSampler] No samplers found in comfy.samplers, using fallback")
            return ["euler", "euler_ancestral", "lms", "heun", "dpmpp_2m", "dpmpp_sde"]
            
    except Exception as e:
        print(f"[PresetToKSampler] Error detecting samplers: {e}")
        return ["euler", "euler_ancestral", "lms", "heun", "dpmpp_2m", "dpmpp_sde"]

def _get_scheduler_choices():
    """Get available schedulers from comfy.samplers (source of truth)"""
    try:
        import comfy.samplers as comfy_samplers
        
        # Get the global registry of registered schedulers
        schedulers = getattr(comfy.samplers, "SCHEDULERS", [])
        
        # Remove None/empty values and duplicates
        schedulers = [s for s in schedulers if s]
        schedulers = sorted(set(schedulers))
        
        if schedulers:
            print(f"[PresetToKSampler] Found {len(schedulers)} schedulers from comfy.samplers")
            return schedulers
        else:
            print("[PresetToKSampler] No schedulers found in comfy.samplers, using fallback")
            return ["normal", "karras", "exponential", "sgm_uniform"]
            
    except Exception as e:
        print(f"[PresetToKSampler] Error detecting schedulers: {e}")
        return ["normal", "karras", "exponential", "sgm_uniform"]

class PresetToKSampler:
    """
    Converts preset parameters to KSampler-compatible inputs
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get dynamic sampler and scheduler choices
        sampler_choices = _get_sampler_choices()
        scheduler_choices = _get_scheduler_choices()
        
        return {
            "required": {
                "sampler_name": (sampler_choices, {"default": "euler"}),
                "scheduler": (scheduler_choices, {"default": "normal"}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.1, "max": 20.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = (
        "STRING",   # sampler_name
        "STRING",   # scheduler
        "INT",      # steps
        "FLOAT",    # cfg
        "INT",      # seed
    )
    RETURN_NAMES = (
        "sampler_name",
        "scheduler",
        "steps",
        "cfg",
        "seed"
    )
    FUNCTION = "convert"
    CATEGORY = "🤖 Model Preset Pilot"

    def convert(self, sampler_name="euler", scheduler="normal", steps=28, cfg=5.5, seed=0):
        """
        Simply pass through the values - this node exists to provide COMBO inputs
        that can be connected to KSampler
        """
        return (sampler_name, scheduler, steps, cfg, seed)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PresetToKSampler": PresetToKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PresetToKSampler": "🔗 Preset to KSampler",
}
