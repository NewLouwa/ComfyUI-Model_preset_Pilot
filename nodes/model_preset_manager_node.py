# Model Preset Manager Node
# Unified ComfyUI node for creating, loading, and managing model presets with preview

import os
import json
import hashlib
import shutil
import base64
import mimetypes
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

import torch
from PIL import Image
import numpy as np

import comfy.utils
import nodes
import folder_paths
from .storage_manager import create_preset, get_preset, get_all_presets, save_preview_image, load_preview_image

# Try to read available samplers/schedulers from Comfy's KSampler
try:
    from nodes import KSampler
    SAMPLER_CHOICES = getattr(KSampler, "SAMPLERS", ["euler", "euler_ancestral", "lms", "heun", "dpmpp_2m", "dpmpp_sde"])
    SCHEDULER_CHOICES = getattr(KSampler, "SCHEDULERS", ["normal", "karras", "exponential", "sgm_uniform"])
except Exception:
    SAMPLER_CHOICES = ["euler", "euler_ancestral", "lms", "heun", "dpmpp_2m", "dpmpp_sde"]
    SCHEDULER_CHOICES = ["normal", "karras", "exponential", "sgm_uniform"]

# Data directory for default assets and templates
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULTS_DIR = os.path.join(DATA_DIR, "defaults")
ASSETS_DIR = os.path.join(DATA_DIR, "assets")
PRESET_DIR = os.path.join(DATA_DIR, "presets")
PREVIEW_DIR = os.path.join(DATA_DIR, "previews")

os.makedirs(DEFAULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(PRESET_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Register our custom API endpoint for file uploads
def api_load_preview_image(json_data):
    """API endpoint to handle preview image uploads"""
    try:
        if "image_data" not in json_data or "model_id" not in json_data:
            return {"error": "Missing required fields"}
        
        # Decode base64 image
        image_data = json_data["image_data"]
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Save to preview directory
        model_id = json_data["model_id"]
        preview_path = os.path.join(PREVIEW_DIR, f"{model_id}_preview.png")
        
        with open(preview_path, "wb") as f:
            f.write(image_bytes)
        
        return {"success": True, "path": preview_path}
    except Exception as e:
        return {"error": str(e)}

# API endpoint registration removed for compatibility
# server.PromptServer.instance.app.post("/model_preset_pilot/upload_preview")(api_load_preview_image)

def _pil_to_image_tensor(pil: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor format [1,H,W,C]"""
    img = pil.convert("RGB")
    arr = np.array(img)  # numpy array uint8
    t = torch.from_numpy(arr).float() / 255.0
    return t.unsqueeze(0)  # batch = 1

def _image_tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert ComfyUI tensor [1,H,W,C] to PIL Image"""
    t = t[0].clamp(0, 1)
    arr = (t.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)

def _get_all_preset_choices():
    """Get all available presets as choices for the dropdown"""
    try:
        # Direct path to presets directory
        presets_dir = os.path.join(DATA_DIR, "presets", "models")
        
        print(f"Looking for presets directory: {presets_dir}")
        
        if not os.path.exists(presets_dir):
            print(f"Presets directory does not exist: {presets_dir}")
            return ["new_preset"]
        
        choices = ["new_preset"]
        
        # Scan presets directory directly
        print(f"Scanning presets directory: {presets_dir}")
        for model_id in os.listdir(presets_dir):
            model_path = os.path.join(presets_dir, model_id)
            if not os.path.isdir(model_path):
                continue
                
            print(f"Checking model: {model_id}")
            
            # Scan presets for this model
            for preset_id in os.listdir(model_path):
                preset_path = os.path.join(model_path, preset_id)
                if os.path.isdir(preset_path):
                    # Check if preset.json exists
                    preset_file = os.path.join(preset_path, "preset.json")
                    if os.path.exists(preset_file):
                        choice = f"{model_id}/{preset_id}"
                        choices.append(choice)
                        print(f"Found preset: {choice}")
        
        print(f"Total preset choices: {choices}")
        return choices
    except Exception as e:
        print(f"Error loading preset choices: {e}")
        return ["new_preset"]

# Initialize preset choices at module level
_preset_choices = ["new_preset"]

def _get_default_preview_image() -> torch.Tensor:
    """Get a default preview image when no preset preview is available"""
    return _create_fallback_image()

def _create_fallback_image() -> torch.Tensor:
    """Create a fallback image when no preview is available"""
    try:
        # Try to load one of the default robot images
        default_images = [
            "NothingHere_Robot.png",
            "NothingHere_Robot2.png", 
            "NothingHere_Robot3.png",
            "NothingHere_Robot4.png"
        ]
        
        for img_name in default_images:
            img_path = os.path.join(DEFAULTS_DIR, img_name)
            if os.path.exists(img_path):
                try:
                    pil = Image.open(img_path)
                    return _pil_to_image_tensor(pil)
                except Exception as e:
                    print(f"Warning: Could not load default image {img_name}: {e}")
                    continue
        
        # If no default images found, create a simple colored rectangle
        pil = Image.new("RGB", (512, 512), (64, 64, 64))
        return _pil_to_image_tensor(pil)
        
    except Exception as e:
        print(f"Warning: Could not create fallback image: {e}")
        # Return a black image as last resort
        return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

def _load_preset_preview_image(model_id: str, preset_id: str) -> torch.Tensor:
    """Load preview image for a specific preset, or return default if not found"""
    from .storage_manager import _get_preset_preview_file
    
    preview_file = _get_preset_preview_file(model_id, preset_id)
    
    if os.path.exists(preview_file):
        try:
            pil = Image.open(preview_file)
            return _pil_to_image_tensor(pil)
        except Exception as e:
            print(f"Warning: Could not load preset preview image: {e}")
    
    # Return default image if preset preview not found
    return _get_default_preview_image()

class ModelPresetManager:
    """
    Unified Model Preset Manager - Create, load, and manage model presets with preview
    Combines functionality of PresetCreator and ModelPresetLoader
    """
    
    aux_id = "NewLouwa/ComfyUI-Model_preset_Pilot"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Update preset choices dynamically
        global _preset_choices
        _preset_choices = _get_all_preset_choices()
        
        return {
            "required": {
                "preset_name": (_preset_choices,),
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
            },
            "optional": {
                "save_preset": ("BOOLEAN", {"default": False}),
                "new_preset_name": ("STRING", {"default": "preset_XXX"}),
                "sampler_name": (SAMPLER_CHOICES, {"default": "euler"}),
                "scheduler": (SCHEDULER_CHOICES, {"default": "normal"}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.1, "max": 20.0}),
                "clip_skip": ("INT", {"default": 0, "min": 0, "max": 12}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = (
        "STRING",   # preset data as string
        "IMAGE",    # preview image
        "STRING",   # status message
    )
    RETURN_NAMES = (
        "preset_data",
        "preview_image", 
        "status"
    )
    FUNCTION = "run"
    CATEGORY = "🤖 Model Preset Pilot"

    def run(self, preset_name="none", model_name="", save_preset=False, new_preset_name="",
            sampler_name="euler", scheduler="normal", steps=28, cfg=5.5,
            clip_skip=0, width=1024, height=1024, seed=0, unique_id=None, extra_pnginfo=None):
        
        # Handle preset loading
        if preset_name and preset_name != "new_preset":
            try:
                # Parse the preset_name format: "model_name/preset_name"
                if "/" in preset_name:
                    model_display_name, preset_id = preset_name.split("/", 1)
                    
                    # Find the actual model_id from the model_name
                    from .storage_manager import _load_model_database
                    db = _load_model_database()
                    
                    # Find the model_id that matches this model_name
                    actual_model_id = None
                    for mid, model_entry in db.get("models", {}).items():
                        checkpoint_name = model_entry.get("checkpoint_name", "")
                        if checkpoint_name:
                            # Extract filename without path and extension
                            import os
                            base_name = os.path.splitext(os.path.basename(checkpoint_name))[0]
                            if base_name == model_display_name:
                                actual_model_id = mid
                                break
                    
                    if not actual_model_id:
                        print(f"Warning: Could not find model ID for {model_display_name}")
                        default_image = _get_default_preview_image()
                        return ("Error: Model not found", default_image, "❌ Model not found")
                    
                    # Load the preset data
                    preset_data = get_preset(actual_model_id, preset_id)
                    
                    if not preset_data:
                        print(f"Warning: Could not load preset {preset_id} for model {actual_model_id}")
                        default_image = _get_default_preview_image()
                        return ("Error: Preset not found", default_image, "❌ Preset not found")
                    
                    print(f"Loaded preset '{preset_id}' for model '{model_display_name}'")
                    print(f"Preset data: {preset_data}")
                    
                    # Load preview image for this preset
                    preview_image = _load_preset_preview_image(actual_model_id, preset_id)
                    
                    # Display preview image internally
                    try:
                        import comfy.utils
                        if preview_image is not None:
                            img_array = (preview_image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                            if hasattr(comfy.utils, 'save_images'):
                                comfy.utils.save_images(img_array, filename_prefix="preset_preview")
                            else:
                                print(f"Preview: Displaying preset image {img_array.shape}")
                    except Exception as e:
                        print(f"Warning: Could not display preview image: {e}")
                    
                    # Return preset data as string and preview image
                    import json
                    preset_json = json.dumps(preset_data, indent=2)
                    return (preset_json, preview_image, f"✅ Loaded preset: {preset_id}")
                            
            except Exception as e:
                print(f"Warning: Could not load preset '{preset_name}': {e}")
                # Return default image and error message
                default_image = _get_default_preview_image()
                
                # Display default image internally
                try:
                    import comfy.utils
                    if default_image is not None:
                        img_array = (default_image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                        if hasattr(comfy.utils, 'save_images'):
                            comfy.utils.save_images(img_array, filename_prefix="preset_error")
                        else:
                            print(f"Preview: Displaying error image {img_array.shape}")
                except Exception as display_e:
                    print(f"Warning: Could not display error image: {display_e}")
                
                return ("Error loading preset", default_image, f"❌ Error: {str(e)}")
        
        # Handle preset saving
        if save_preset and model_name:
            try:
                # Generate model ID from model name
                import os
                model_id = os.path.splitext(os.path.basename(model_name))[0]
                
                if preset_name == "new_preset":
                    # Creating a new preset
                    preset_id = new_preset_name if new_preset_name else f"preset_{int(datetime.now().timestamp())}"
                    preset_display_name = new_preset_name if new_preset_name else f"Preset for {model_id}"
                    action = "created"
                else:
                    # Modifying existing preset
                    if "/" in preset_name:
                        _, preset_id = preset_name.split("/", 1)
                    else:
                        preset_id = preset_name
                    preset_display_name = preset_id
                    action = "updated"
                
                preset_data = {
                    "id": preset_id,
                    "name": preset_display_name,
                    "description": f"Preset {action} for {model_id}",
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "steps": steps,
                    "cfg": cfg,
                    "clip_skip": clip_skip,
                    "width": width,
                    "height": height,
                    "seed": seed,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "tags": ["user_created"]
                }
                
                # Save the preset
                create_preset(model_id, preset_data)
                
                # Update model database
                from .storage_manager import _load_model_database, _save_model_database
                db = _load_model_database()
                if "models" not in db:
                    db["models"] = {}
                
                db["models"][model_id] = {
                    "checkpoint_name": model_name,
                    "display_name": model_id,
                    "created_at": datetime.now().isoformat()
                }
                _save_model_database(db)
                
                # Display success message
                status_msg = f"✅ Preset {action}! 📝 Name: {preset_data['name']} 📁 Location: {model_id}/{preset_data['id']} ⚙️ Settings: {sampler_name}, {scheduler}, {steps} steps"
                print(status_msg)
                
                # Return default image and success message
                import json
                default_image = _get_default_preview_image()
                return (json.dumps(preset_data, indent=2), default_image, status_msg)
                
            except Exception as e:
                error_msg = f"❌ Error saving preset: {str(e)}"
                print(error_msg)
                default_image = _get_default_preview_image()
                return ("Error saving preset", default_image, error_msg)
        
        # No preset selected and not saving
        print("No preset selected and not saving")
        default_image = _get_default_preview_image()
        
        # Display default image internally
        try:
            import comfy.utils
            if default_image is not None:
                img_array = (default_image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                if hasattr(comfy.utils, 'save_images'):
                    comfy.utils.save_images(img_array, filename_prefix="preset_default")
                else:
                    print(f"Preview: Displaying default image {img_array.shape}")
        except Exception as display_e:
            print(f"Warning: Could not display default image: {display_e}")
        
        return ("No preset data", default_image, "📋 Select a preset to load, or 'new_preset' + Save Preset to create")

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ModelPresetManager": ModelPresetManager,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelPresetManager": "🎛️ Model Preset Manager",
}
