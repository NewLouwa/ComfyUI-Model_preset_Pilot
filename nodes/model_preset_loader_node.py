# ModelPresetPilot Node
# Custom ComfyUI node for managing model presets with preview

import os
import json
import hashlib
import shutil
import base64
import mimetypes
from typing import Any, Dict, Optional, List, Tuple

import torch
from PIL import Image
import numpy as np

import comfy.utils
import nodes
import folder_paths
import server
from server import PromptServer

# Try to read available samplers/schedulers from Comfy's KSampler;
# fall back to a safe list if the attributes aren't exposed.
try:
    from nodes import KSampler
    SAMPLER_CHOICES = getattr(KSampler, "SAMPLERS", ["euler", "euler_ancestral", "lms", "heun", "dpmpp_2m", "dpmpp_sde"])
    SCHEDULER_CHOICES = getattr(KSampler, "SCHEDULERS", ["normal", "karras", "exponential", "sgm_uniform"])
except Exception:
    SAMPLER_CHOICES = ["euler", "euler_ancestral", "lms", "heun", "dpmpp_2m", "dpmpp_sde"]
    SCHEDULER_CHOICES = ["normal", "karras", "exponential", "sgm_uniform"]

# Where we'll persist presets & previews
COMFY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRESET_DIR = os.path.join(COMFY_ROOT, "user", "model_presets")
PREVIEW_DIR = os.path.join(PRESET_DIR, "previews")

# Data directory for default assets and templates
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEFAULTS_DIR = os.path.join(DATA_DIR, "defaults")
ASSETS_DIR = os.path.join(DATA_DIR, "assets")

os.makedirs(PRESET_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(DEFAULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

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
        1) model.model_name if present (or inner model_name/ckpt_path)
        2) ckpt_name string
        3) hashed fallback
    """
    if model is not None:
        name = getattr(model, "model_name", None)
        if not name:
            inner = getattr(model, "model", None)
            name = getattr(inner, "model_name", None) or getattr(inner, "ckpt_path", None)
        if name:
            return _sanitize_filename(str(name))
    if ckpt_name:
        return _sanitize_filename(str(ckpt_name))
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
    
    # Use realistic preset as default if available
    if "realistic" in default_presets:
        return default_presets["realistic"]
    
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
    arr = comfy.utils.to_uint8(img)  # returns np.uint8 array (H, W, C)
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
        return float("NaN")  # Always update to handle button clicks
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True
        
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
            textDisplay.style.height = "80px";
            textDisplay.style.padding = "8px";
            textDisplay.style.backgroundColor = "rgba(0,0,0,0.7)";
            textDisplay.style.borderRadius = "4px";
            textDisplay.style.fontSize = "11px";
            textDisplay.style.fontFamily = "monospace";
            textDisplay.style.color = "#ccc";
            textDisplay.style.border = "1px solid #444";
            textDisplay.style.overflow = "hidden";
            textDisplay.style.display = "flex";
            textDisplay.style.alignItems = "center";
            textDisplay.style.justifyContent = "center";
            textDisplay.innerHTML = "Model: Not connected<br>ID: Unknown";
            
            // Create image preview canvas (positioned at bottom)
            const previewCanvas = document.createElement("canvas");
            previewCanvas.width = 150;
            previewCanvas.height = 150;
            previewCanvas.style.position = "absolute";
            previewCanvas.style.bottom = "10px";
            previewCanvas.style.left = "10px";
            previewCanvas.style.right = "10px";
            previewCanvas.style.border = "2px solid #333";
            previewCanvas.style.borderRadius = "8px";
            previewCanvas.style.backgroundColor = "#1a1a1a";
            previewCanvas.style.display = "block";
            previewCanvas.style.margin = "0 auto";
            
            // Add elements to the node
            const nodeContainer = node.widgets[0].options.el.parentElement.parentElement;
            nodeContainer.style.position = "relative";
            nodeContainer.style.minHeight = "300px";
            nodeContainer.appendChild(textDisplay);
            nodeContainer.appendChild(previewCanvas);
            
            // Function to update the display with default preview
            function updateDisplay() {
                const ctx = previewCanvas.getContext("2d");
                
                // Create a default preview image with gradient background
                const gradient = ctx.createLinearGradient(0, 0, 150, 150);
                gradient.addColorStop(0, "#2a2a2a");
                gradient.addColorStop(1, "#1a1a1a");
                ctx.fillStyle = gradient;
                ctx.fillRect(0, 0, 150, 150);
                
                // Add border
                ctx.strokeStyle = "#444";
                ctx.lineWidth = 2;
                ctx.strokeRect(1, 1, 148, 148);
                
                // Add default preview icon
                ctx.fillStyle = "#666";
                ctx.font = "bold 14px Arial";
                ctx.textAlign = "center";
                ctx.fillText("🖼️", 75, 70);
                ctx.font = "10px Arial";
                ctx.fillText("Preview", 75, 90);
                ctx.fillText("Image", 75, 105);
            }
            
            // Initialize display
            updateDisplay();
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
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = (
        "MODEL",    # model (pass through)
        "IMAGE",    # preview image
        "STRING",   # model_info (text display)
    )
    RETURN_NAMES = (
        "model", "preview", "model_info"
    )
    FUNCTION = "run"
    CATEGORY = "🤖 Model Preset Pilot"

    def _empty_image(self):
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)
    
    def _create_default_preview(self, width=150, height=150):
        """Create a default preview image with a nice gradient and icon"""
        # Create a gradient background
        img = Image.new('RGB', (width, height), '#1a1a1a')
        
        # Create a simple default preview with text
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        
        # Add gradient effect
        for y in range(height):
            alpha = int(255 * (1 - y / height))
            color = (int(26 + alpha * 0.1), int(26 + alpha * 0.1), int(26 + alpha * 0.1))
            draw.line([(0, y), (width, y)], fill=color)
        
        # Add border
        draw.rectangle([0, 0, width-1, height-1], outline='#444', width=2)
        
        # Add preview text
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None
            
        # Add icon and text
        draw.text((width//2, height//2 - 20), "🖼️", font=font, anchor="mm", fill="#666")
        draw.text((width//2, height//2 + 10), "Preview", font=font, anchor="mm", fill="#666")
        draw.text((width//2, height//2 + 25), "Image", font=font, anchor="mm", fill="#666")
        
        return _pil_to_image_tensor(img)

    def _save_image_as_preview(self, image_tensor: torch.Tensor, preview_file: str) -> None:
        """Save the given image tensor as a preview image"""
        os.makedirs(os.path.dirname(preview_file), exist_ok=True)
        
        # Create a backup of the original image if it exists
        if os.path.exists(preview_file):
            backup_file = f"{preview_file}.bak"
            try:
                shutil.copy2(preview_file, backup_file)
            except Exception as e:
                print(f"Warning: Could not create backup of preview image: {e}")
                
        # Save the new preview image
        _image_tensor_to_pil(image_tensor).save(preview_file)
        
    def run(self, model, image=None, unique_id=None, extra_pnginfo=None):
        # Get model identifier
        model_id = _model_identifier(model, None)
        
        # Create model info text
        model_name = getattr(model, "model_name", "Unknown Model")
        model_info = f"Model: {model_name}\nID: {model_id}"
        
        # Handle preview image
        preview_tensor = None
        
        if image is not None and image.shape[0] > 0:
            # Use provided image as preview
            preview_tensor = image
        else:
            # Try to load existing preview
            preview_file = _preview_path(model_id)
            if os.path.exists(preview_file):
                try:
                    preview_tensor = _pil_to_image_tensor(Image.open(preview_file).convert("RGB"))
                except Exception:
                    preview_tensor = None
            
            # Fallback to default preview image if no preview available
            if preview_tensor is None:
                preview_tensor = self._create_default_preview()

        return (model, preview_tensor, model_info)
