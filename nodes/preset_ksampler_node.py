"""
Preset KSampler Node
A specialized KSampler that can be directly connected to ModelPresetManager
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
            print(f"[PresetKSampler] Found {len(valid_samplers)} valid samplers from comfy.samplers")
            return valid_samplers
        elif samplers:
            print(f"[PresetKSampler] Found {len(samplers)} samplers from comfy.samplers (not all verified)")
            return samplers
        else:
            print("[PresetKSampler] No samplers found in comfy.samplers, using fallback")
            return ["euler", "euler_ancestral", "lms", "heun", "dpmpp_2m", "dpmpp_sde"]
            
    except Exception as e:
        print(f"[PresetKSampler] Error detecting samplers: {e}")
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
            print(f"[PresetKSampler] Found {len(schedulers)} schedulers from comfy.samplers")
            return schedulers
        else:
            print("[PresetKSampler] No schedulers found in comfy.samplers, using fallback")
            return ["normal", "karras", "exponential", "sgm_uniform"]
            
    except Exception as e:
        print(f"[PresetKSampler] Error detecting schedulers: {e}")
        return ["normal", "karras", "exponential", "sgm_uniform"]

class PresetKSampler:
    """
    Specialized KSampler that can be connected to ModelPresetManager
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get dynamic sampler and scheduler choices
        sampler_choices = _get_sampler_choices()
        scheduler_choices = _get_scheduler_choices()
        
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "sampler_name": (sampler_choices, {"default": "euler"}),
                "scheduler": (scheduler_choices, {"default": "normal"}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.1, "max": 20.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "🤖 Model Preset Pilot"

    def sample(self, model, positive, negative, latent_image, sampler_name="euler", 
               scheduler="normal", steps=28, cfg=5.5, seed=0):
        """
        Sample using the provided parameters
        """
        try:
            # Import the KSampler from ComfyUI
            from nodes import KSampler
            
            # Create a KSampler instance and call its sample method
            ksampler = KSampler()
            return ksampler.sample(model, positive, negative, latent_image, 
                                 sampler_name, scheduler, steps, cfg, seed)
            
        except Exception as e:
            print(f"[PresetKSampler] Error during sampling: {e}")
            # Return the input latent as fallback
            return (latent_image,)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PresetKSampler": PresetKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PresetKSampler": "🎲 Preset KSampler",
}
