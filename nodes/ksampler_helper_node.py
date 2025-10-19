"""
KSampler Helper Node
Converts string values to COMBO values for KSampler compatibility
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
            print(f"[KSamplerHelper] Found {len(valid_samplers)} valid samplers from comfy.samplers")
            return valid_samplers
        elif samplers:
            print(f"[KSamplerHelper] Found {len(samplers)} samplers from comfy.samplers (not all verified)")
            return samplers
        else:
            print("[KSamplerHelper] No samplers found in comfy.samplers, using fallback")
            return ["euler", "euler_ancestral", "lms", "heun", "dpmpp_2m", "dpmpp_sde"]
            
    except Exception as e:
        print(f"[KSamplerHelper] Error detecting samplers: {e}")
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
            print(f"[KSamplerHelper] Found {len(schedulers)} schedulers from comfy.samplers")
            return schedulers
        else:
            print("[KSamplerHelper] No schedulers found in comfy.samplers, using fallback")
            return ["normal", "karras", "exponential", "sgm_uniform"]
            
    except Exception as e:
        print(f"[KSamplerHelper] Error detecting schedulers: {e}")
        return ["normal", "karras", "exponential", "sgm_uniform"]

class KSamplerHelper:
    """
    Helper node to convert string values to COMBO values for KSampler
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
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = (
        "STRING",   # sampler_name
        "STRING",   # scheduler
        "INT",      # steps
        "FLOAT",    # cfg
        "INT",      # width
        "INT",      # height
        "INT",      # seed
    )
    RETURN_NAMES = (
        "sampler_name",
        "scheduler",
        "steps",
        "cfg",
        "width",
        "height",
        "seed"
    )
    FUNCTION = "run"
    CATEGORY = "🤖 Model Preset Pilot"

    def run(self, sampler_name="euler", scheduler="normal", steps=28, cfg=5.5,
            width=1024, height=1024, seed=0):
        """
        Simply pass through the values - this node exists to provide COMBO inputs
        that can be connected to KSampler
        """
        return (sampler_name, scheduler, steps, cfg, width, height, seed)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "KSamplerHelper": KSamplerHelper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KSamplerHelper": "🔧 KSampler Helper",
}
