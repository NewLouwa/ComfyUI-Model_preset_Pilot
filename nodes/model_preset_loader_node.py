# ModelPresetPilot Node
# Custom ComfyUI node for managing model presets with preview

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
import server
from server import PromptServer
from .storage_manager import create_preset, get_preset, get_all_presets, save_preview_image, load_preview_image

# Try to read available samplers/schedulers from Comfy's KSampler;
# fall back to a safe list if the attributes aren't exposed.
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
            return {"success": False, "error": "Missing required data"}
            
        model_id = json_data["model_id"]
        image_data = json_data["image_data"]
        
        # Decode base64 image
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
            
        image_bytes = base64.b64decode(image_data)
        
        # Save to preview directory
        preview_path = _preview_path(model_id)
        
        # Create backup if file exists
        if os.path.exists(preview_path):
            backup_path = f"{preview_path}.bak"
            try:
                shutil.copy2(preview_path, backup_path)
            except Exception as e:
                print(f"Warning: Failed to create backup: {e}")
        
        # Save new image
        with open(preview_path, "wb") as f:
            f.write(image_bytes)
            
        # Get image info
        info = _get_preview_info(model_id)
        
        return {
            "success": True,
            "message": f"Preview image saved for model: {model_id}",
            "path": preview_path,
            "dimensions": info.get("dimensions"),
            "size": info.get("size")
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# Register the API endpoint
try:
    # Try the new ComfyUI API method first
    if hasattr(PromptServer.instance, 'add_api_route'):
        PromptServer.instance.add_api_route("/model_preset_pilot/upload_preview", api_load_preview_image, methods=["POST"])
    else:
        # Fallback to the older method
        PromptServer.instance.app.add_route("/model_preset_pilot/upload_preview", api_load_preview_image, methods=["POST"])
except Exception as e:
    print(f"Warning: Could not register API route: {e}")


def _sanitize_filename(name: str) -> str:
    keep = "-_.() "
    safe = "".join(c for c in name if c.isalnum() or c in keep)
    safe = "_".join(safe.split())
    return safe[:128] or "model"


def _model_identifier(model: Optional[Any], ckpt_name: Optional[str]) -> str:
    """
    Build a stable identifier for the model to name files.
    Priority:
        1) Try to get model name from ComfyUI's model loading system
        2) model.model_name if present (or inner model_name/ckpt_path)
        3) ckpt_name string
        4) Try to extract filename from model object attributes
        5) hashed fallback
    """
    if model is not None:
        # First, try to get the model name from ComfyUI's model loading system
        try:
            # Check if we can get the model name from the model's internal structure
            if hasattr(model, 'model') and hasattr(model.model, '__class__'):
                model_class_name = model.model.__class__.__name__
                
                # Try to get the model name from the model's config or state
                if hasattr(model.model, 'config') and hasattr(model.model.config, 'name'):
                    name = model.model.config.name
                elif hasattr(model.model, 'name'):
                    name = model.model.name
                else:
                    name = None
        except Exception:
            name = None
        
        # Try model.model_name first
        if not name:
            name = getattr(model, "model_name", None)
        
        if not name:
            # Try inner model attributes
            inner = getattr(model, "model", None)
            if inner:
                name = getattr(inner, "model_name", None) or getattr(inner, "ckpt_path", None)
        
        # If still no name, try other common attributes
        if not name:
            # Try common model attributes that might contain filename
            for attr in ["filename", "file_name", "path", "ckpt_path", "name", "config_path"]:
                if hasattr(model, attr):
                    attr_value = getattr(model, attr)
                    # Skip methods and None values, only use strings
                    if attr_value and str(attr_value) != "None" and not callable(attr_value) and isinstance(attr_value, str):
                        name = attr_value
                        break
                # Also check inner model
                if inner and hasattr(inner, attr):
                    attr_value = getattr(inner, attr)
                    if attr_value and str(attr_value) != "None" and not callable(attr_value) and isinstance(attr_value, str):
                        name = attr_value
                        break
        
        # If we found a name, extract just the filename without extension
        if name:
            name_str = str(name)
            # Extract filename from path if it's a full path
            if "/" in name_str or "\\" in name_str:
                name_str = os.path.basename(name_str)
            # Remove extension
            name_str = os.path.splitext(name_str)[0]
            return _sanitize_filename(name_str)
    
    if ckpt_name:
        return _sanitize_filename(str(ckpt_name))
    
    # Use a more stable hash based on model structure rather than object repr
    try:
        if model is not None:
            # Try to create a stable hash based on model structure
            model_info = {
                'type': str(type(model)),
                'class': str(model.__class__.__name__) if hasattr(model, '__class__') else 'unknown'
            }
            
            # Add inner model info if available
            if hasattr(model, 'model') and model.model is not None:
                model_info['inner_type'] = str(type(model.model))
                if hasattr(model.model, '__class__'):
                    model_info['inner_class'] = str(model.model.__class__.__name__)
            
            # Create hash from stable model info
            model_str = str(sorted(model_info.items()))
            stable_hash = hashlib.sha256(model_str.encode()).hexdigest()[:10]
            return f"model_{stable_hash}"
    except Exception:
        pass
    
    return f"unknown_{hashlib.sha256(repr(model).encode()).hexdigest()[:10]}"


def _preset_path(model_id: str) -> str:
    return os.path.join(PRESET_DIR, f"{model_id}.json")


def _preview_path(model_id: str) -> str:
    return os.path.join(PREVIEW_DIR, f"{model_id}.png")

def _get_preview_info(model_id: str) -> Dict[str, Any]:
    """Get information about the preview image for a model"""
    preview_file = _preview_path(model_id)
    info = {
        "exists": False,
        "path": preview_file,
        "size": None,
        "dimensions": None,
        "last_modified": None,
    }
    
    if os.path.exists(preview_file):
        info["exists"] = True
        stat = os.stat(preview_file)
        info["size"] = stat.st_size
        info["last_modified"] = stat.st_mtime
        
        try:
            with Image.open(preview_file) as img:
                info["dimensions"] = img.size
        except Exception:
            pass
                
    return info


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


def _get_default_preview_image() -> torch.Tensor:
    """Get a default preview image from the defaults directory"""
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
    
    # Fallback: create a simple colored image
    return _create_fallback_image()


def _create_fallback_image() -> torch.Tensor:
    """Create a simple fallback image if no defaults are available"""
    # Create a simple 512x512 image with a gradient using PIL
    pil = Image.new("RGB", (512, 512), "gray")
    return _pil_to_image_tensor(pil)


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


def _load_default_templates() -> Dict[str, Any]:
    """Load default preset templates from data directory"""
    templates_path = os.path.join(DEFAULTS_DIR, "preset_templates.json")
    if os.path.exists(templates_path):
        try:
            with open(templates_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load default templates: {e}")
    return {"default_presets": {}}

def _load_preset(model_id: str) -> Dict[str, Any]:
    path = _preset_path(model_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Try to load from default templates
    templates = _load_default_templates()
    default_presets = templates.get("default_presets", {})
    
    # Use realistic preset as default if available, otherwise use none
    if "realistic" in default_presets:
        return default_presets["realistic"]
    elif "none" in default_presets:
        return default_presets["none"]
    
    # Fallback to sensible defaults
    return {
        "sampler_name": "dpmpp_2m" if "dpmpp_2m" in SAMPLER_CHOICES else SAMPLER_CHOICES[0],
        "scheduler": "karras" if "karras" in SCHEDULER_CHOICES else SCHEDULER_CHOICES[0],
        "steps": 28,
        "cfg": 5.5,
        "clip_skip": 0,
        "width": 1024,
        "height": 1024,
        "seed": 0,
    }


def _save_preset(model_id: str, data: Dict[str, Any]) -> None:
    with open(_preset_path(model_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _pil_to_image_tensor(pil: Image.Image) -> torch.Tensor:
    # ComfyUI expects float32 0..1, shape [B, H, W, C]
    img = pil.convert("RGB")
    arr = np.array(img)  # numpy array uint8
    t = torch.from_numpy(arr).float() / 255.0
    return t.unsqueeze(0)


def _image_tensor_to_pil(t: torch.Tensor) -> Image.Image:
    # t: [B, H, W, C], 0..1 float
    t = t[0].clamp(0, 1)
    arr = (t.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


class ModelPresetLoader:
    """
    Model Preset Loader - loads and displays model presets with preview
    """
    
    aux_id = "NewLouwa/ComfyUI-Model_preset_Pilot"
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force update to refresh preset choices
        return float("NaN")  # Always update
    
    
        
    @classmethod
    def get_js(cls):
        # Add custom JavaScript for the simplified UI
        js = """
        // Add custom JavaScript for the Model Preset Loader node
        function setupModelPresetLoaderUI(node) {
            // Create text display area (positioned at top)
            const textDisplay = document.createElement("div");
            textDisplay.className = "model-info-display";
            textDisplay.style.position = "absolute";
            textDisplay.style.top = "60px";
            textDisplay.style.left = "10px";
            textDisplay.style.right = "10px";
            textDisplay.style.height = "50px";
            textDisplay.style.padding = "6px";
            textDisplay.style.backgroundColor = "rgba(0,0,0,0.7)";
            textDisplay.style.borderRadius = "4px";
            textDisplay.style.fontSize = "10px";
            textDisplay.style.fontFamily = "monospace";
            textDisplay.style.color = "#ccc";
            textDisplay.style.border = "1px solid #444";
            textDisplay.style.overflow = "hidden";
            textDisplay.style.display = "flex";
            textDisplay.style.alignItems = "center";
            textDisplay.style.justifyContent = "center";
            textDisplay.innerHTML = "Model: Not connected<br>ID: Unknown<br>Presets: 0 available";
            
            // Create image preview canvas (positioned at bottom)
            const previewCanvas = document.createElement("canvas");
            previewCanvas.width = 120;
            previewCanvas.height = 120;
            previewCanvas.style.position = "absolute";
            previewCanvas.style.bottom = "10px";
            previewCanvas.style.left = "10px";
            previewCanvas.style.right = "10px";
            previewCanvas.style.border = "2px solid #333";
            previewCanvas.style.borderRadius = "6px";
            previewCanvas.style.backgroundColor = "#1a1a1a";
            previewCanvas.style.display = "block";
            previewCanvas.style.margin = "0 auto";
            
            // Add elements to the node
            const nodeContainer = node.widgets[0].options.el.parentElement.parentElement;
            nodeContainer.style.position = "relative";
            nodeContainer.style.minHeight = "250px";
            nodeContainer.appendChild(textDisplay);
            nodeContainer.appendChild(previewCanvas);
            
            // Function to update the display with default preview
            function updateDisplay() {
                const ctx = previewCanvas.getContext("2d");
                
                // Create a default preview image with gradient background
                const gradient = ctx.createLinearGradient(0, 0, 120, 120);
                gradient.addColorStop(0, "#2a2a2a");
                gradient.addColorStop(1, "#1a1a1a");
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, 120, 120);
                
                // Add border
                ctx.strokeStyle = "#444";
                ctx.lineWidth = 2;
                ctx.strokeRect(1, 1, 118, 118);
                
                // Add default preview icon
                ctx.fillStyle = "#666";
                ctx.font = "bold 12px Arial";
                ctx.textAlign = "center";
                ctx.fillText("🖼️", 60, 55);
                ctx.font = "9px Arial";
                ctx.fillText("Preview", 60, 75);
                ctx.fillText("Image", 60, 90);
            }
            
            // Function to update preset choices dynamically
            function updatePresetChoices() {
                // Find the preset_name widget and update its choices
                const presetWidget = node.widgets.find(w => w.name === "preset_name");
                if (presetWidget && presetWidget.options) {
                    // This will be handled by the backend when the model changes
                    console.log("Preset choices will be updated by backend");
                }
            }
            
            // Initialize display
            updateDisplay();
            
            // Store functions on the node for external access
            node.updatePresetChoices = updatePresetChoices;
        }
        
        // Register a callback when nodes are added to the graph
        app.registerExtension({
            name: "ModelPresetLoader.UI",
            async nodeCreated(node) {
                if (node.comfyClass === "ModelPresetLoader") {
                    // Wait for widgets to be created
                    setTimeout(() => setupModelPresetLoaderUI(node), 100);
                }
            }
        });
        """
        return js

    @classmethod
    def INPUT_TYPES(cls):
        # Force refresh of preset choices every time
        preset_choices = cls._get_all_preset_choices()
        print(f"INPUT_TYPES called, found {len(preset_choices)} choices: {preset_choices}")
        
        return {
            "required": {
                "preset_name": (preset_choices,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    
    @classmethod
    def _get_all_preset_choices(cls):
        """Get all available presets across all models, formatted as model_name/preset_name"""
        preset_choices = ["none"]
        
        try:
            # Use absolute path to ensure we find the directory
            models_dir = os.path.abspath(os.path.join(PRESET_DIR, "models"))
            print(f"Looking for models directory: {models_dir}")
            
            if not os.path.exists(models_dir):
                print(f"Models directory not found: {models_dir}")
                # Try alternative path
                alt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "presets", "models"))
                print(f"Trying alternative path: {alt_path}")
                if os.path.exists(alt_path):
                    models_dir = alt_path
                else:
                    return preset_choices
            
            print(f"Scanning models directory: {models_dir}")
            
            # Load model database for better names
            from .storage_manager import _load_model_database
            db = _load_model_database()
            print(f"Model database loaded: {len(db.get('models', {}))} models")
            
            # Scan all model directories
            for model_id in os.listdir(models_dir):
                model_dir = os.path.join(models_dir, model_id)
                if not os.path.isdir(model_dir):
                    continue
                
                print(f"Checking model: {model_id}")
                
                # Get a better display name for the model
                model_entry = db.get("models", {}).get(model_id, {})
                display_name = model_entry.get("checkpoint_name", model_id)
                
                # If we have a checkpoint_name, use the filename without path
                if display_name and display_name != model_id:
                    # Extract just the filename from the full path
                    display_name = os.path.basename(display_name)
                    # Remove extension
                    display_name = os.path.splitext(display_name)[0]
                
                # If it's still a hash, try to make it more readable
                if display_name.startswith("model_") or display_name.startswith("unknown_"):
                    # Try to extract type from model_id or use a generic name
                    if "SDXL" in model_id.upper():
                        display_name = "SDXL_Model"
                    elif "SD" in model_id.upper():
                        display_name = "SD_Model"
                    elif "FLUX" in model_id.upper():
                        display_name = "FLUX_Model"
                    elif "PONY" in model_id.upper():
                        display_name = "PONY_Model"
                    elif "ILLUST" in model_id.upper():
                        display_name = "ILLUST_Model"
                    else:
                        display_name = f"Model_{model_id[-8:]}"  # Use last 8 chars of hash
                
                print(f"Display name: {display_name}")
                
                # Scan all presets for this model
                preset_count = 0
                for preset_name in os.listdir(model_dir):
                    preset_path = os.path.join(model_dir, preset_name)
                    if os.path.isdir(preset_path):
                        # Check if it's a valid preset directory (has preset.json)
                        preset_file = os.path.join(preset_path, "preset.json")
                        if os.path.exists(preset_file):
                            # Format: "display_name/preset_name"
                            preset_choices.append(f"{display_name}/{preset_name}")
                            preset_count += 1
                            print(f"Found preset: {display_name}/{preset_name}")
                
                print(f"Found {preset_count} presets for {display_name}")
            
            print(f"Total preset choices: {preset_choices}")
            return preset_choices if len(preset_choices) > 1 else ["none"]
        except Exception as e:
            print(f"Warning: Could not get preset choices: {e}")
            import traceback
            traceback.print_exc()
            return ["none"]
    

    RETURN_TYPES = (
        "STRING",   # preset data as string
        "IMAGE",    # preview image
    )
    RETURN_NAMES = (
        "preset_data",
        "preview_image"
    )
    FUNCTION = "run"
    CATEGORY = "🤖 Model Preset Pilot"

        
    def run(self, preset_name="none", unique_id=None, extra_pnginfo=None):
        # Load the selected preset if it exists
        if preset_name and preset_name != "none":
            try:
                # Parse the preset_name format: "model_name/preset_name"
                if "/" in preset_name:
                    model_name, preset_id = preset_name.split("/", 1)
                    
                    # Find the actual model_id from the model_name
                    from .storage_manager import _load_model_database
                    db = _load_model_database()
                    
                    # Find the model_id that matches this model_name
                    actual_model_id = None
                    for mid, model_entry in db.get("models", {}).items():
                        checkpoint_name = model_entry.get("checkpoint_name", "")
                        if checkpoint_name:
                            # Extract filename without path and extension
                            clean_name = os.path.splitext(os.path.basename(checkpoint_name))[0]
                            if clean_name == model_name:
                                actual_model_id = mid
                                break
                    
                    if actual_model_id is None:
                        # Fallback: try to find by partial match
                        for mid in os.listdir(os.path.join(PRESET_DIR, "models")):
                            if model_name in mid or mid in model_name:
                                actual_model_id = mid
                                break
                    
                    if actual_model_id is None:
                        print(f"Could not find model for: {model_name}")
                        # Return default image and error message
                        default_image = _get_default_preview_image()
                        return ("No preset data", default_image)
                    
                    preset_model_id = actual_model_id
                else:
                    print(f"Invalid preset format: {preset_name}")
                    # Return default image and error message
                    default_image = _get_default_preview_image()
                    return ("No preset data", default_image)
                
                # Import storage manager functions
                from .storage_manager import get_preset
                preset_data = get_preset(preset_model_id, preset_id)
                
                print(f"Loaded preset '{preset_id}' for model '{model_name}'")
                print(f"Preset data: {preset_data}")
                
                # Load preview image for this preset
                preview_image = _load_preset_preview_image(preset_model_id, preset_id)
                
                # Return preset data as string and preview image
                import json
                preset_json = json.dumps(preset_data, indent=2)
                return (preset_json, preview_image)
                        
            except Exception as e:
                print(f"Warning: Could not load preset '{preset_name}': {e}")
                # Return default image and error message
                default_image = _get_default_preview_image()
                return ("Error loading preset", default_image)
        else:
            print("No preset selected")
            # Return default image and no data message
            default_image = _get_default_preview_image()
            return ("No preset data", default_image)
    
