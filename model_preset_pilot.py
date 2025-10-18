# model_preset_pilot.py
# Custom node for ComfyUI
# ------------------------------------------------------------
# Name: Model Preset Pilot
# Purpose:
#  • Manage per-model presets (sampler, scheduler, steps, cfg, clip_skip, width, height, seed)
#  • Automatically load / save / update these presets
#  • Optionally generate and cache preview images per model
# ------------------------------------------------------------

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
COMFY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRESET_DIR = os.path.join(COMFY_ROOT, "user", "model_presets")
PREVIEW_DIR = os.path.join(PRESET_DIR, "previews")
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
PromptServer.instance.add_api_route("/model_preset_pilot/upload_preview", api_load_preview_image, methods=["POST"])


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


def _load_preset(model_id: str) -> Dict[str, Any]:
    path = _preset_path(model_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Sensible defaults if no preset exists yet
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


class ModelPresetPilot:
    """
    Modes:
      • load:   read preset or create default, optional cached preview
      • save:   overwrite preset with current inputs, optional preview generation
      • update: partial update of existing preset
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
        # Add custom JavaScript to enhance the UI
        js = """
        // Add custom JavaScript for the Model Preset Pilot node
        function setupModelPresetPilotUI(node) {
            // Get widgets
            const loadImageCheckbox = node.widgets.filter(w => w.name === "load_image")[0];
            const loadFromFileCheckbox = node.widgets.filter(w => w.name === "load_preview_from_file")[0];
            if (!loadImageCheckbox || !loadFromFileCheckbox) return;
            
            // Create info display element
            const infoDiv = document.createElement("div");
            infoDiv.className = "model-preset-info";
            infoDiv.style.margin = "10px 0";
            infoDiv.style.padding = "8px";
            infoDiv.style.backgroundColor = "rgba(0,0,0,0.1)";
            infoDiv.style.borderRadius = "4px";
            infoDiv.style.fontSize = "12px";
            infoDiv.innerHTML = "<b>Preview Image:</b> No information available";
            
            // Add it after the checkbox
            const widgetContainer = loadImageCheckbox.options.el.parentElement.parentElement;
            widgetContainer.parentElement.insertBefore(infoDiv, widgetContainer.nextSibling);
            
            // Create file upload button
            const uploadButton = document.createElement("button");
            uploadButton.textContent = "Browse for preview image...";
            uploadButton.style.margin = "10px 0";
            uploadButton.style.padding = "5px 10px";
            uploadButton.style.backgroundColor = "#2a2a2a";
            uploadButton.style.color = "white";
            uploadButton.style.border = "none";
            uploadButton.style.borderRadius = "4px";
            uploadButton.style.cursor = "pointer";
            uploadButton.style.width = "100%";
            
            // Add hover effect
            uploadButton.addEventListener("mouseover", () => {
                uploadButton.style.backgroundColor = "#3a3a3a";
            });
            uploadButton.addEventListener("mouseout", () => {
                uploadButton.style.backgroundColor = "#2a2a2a";
            });
            
            // Add it after the info div
            infoDiv.parentElement.insertBefore(uploadButton, infoDiv.nextSibling);
            
            // Create hidden file input
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.accept = "image/*";
            fileInput.style.display = "none";
            uploadButton.parentElement.appendChild(fileInput);
            
            // Connect button to file input
            uploadButton.addEventListener("click", () => {
                fileInput.click();
            });
            
            // Handle file selection
            fileInput.addEventListener("change", async (event) => {
                if (!event.target.files || !event.target.files[0]) return;
                
                const file = event.target.files[0];
                const reader = new FileReader();
                
                reader.onload = async (e) => {
                    // Get model_id
                    let model_id = "unknown";
                    const modelInput = node.inputs ? node.inputs.find(i => i.name === "model") : null;
                    const ckptInput = node.widgets.filter(w => w.name === "ckpt_name")[0];
                    
                    if (modelInput && modelInput.link) {
                        // We have a model connected, but we can't directly access its ID
                        // We'll use a placeholder and let the backend handle it
                        model_id = "connected_model";
                    } else if (ckptInput && ckptInput.value) {
                        model_id = ckptInput.value;
                    }
                    
                    // Show loading state
                    infoDiv.innerHTML = "<b>Preview Image:</b> Uploading...";
                    uploadButton.disabled = true;
                    uploadButton.textContent = "Uploading...";
                    
                    try {
                        // Send to server
                        const response = await fetch("/model_preset_pilot/upload_preview", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                model_id: model_id,
                                image_data: e.target.result
                            })
                        });
                        
                        const result = await response.json();
                        
                        if (result.success) {
                            infoDiv.innerHTML = `<b>Preview Image:</b> Uploaded successfully<br>
                                               Dimensions: ${result.dimensions ? result.dimensions.join('x') : 'unknown'}<br>
                                               Path: ${result.path}`;
                            // Force node to refresh
                            loadFromFileCheckbox.value = true;
                            node.widgets_values[node.widgets.indexOf(loadFromFileCheckbox)] = true;
                        } else {
                            infoDiv.innerHTML = `<b>Preview Image:</b> Upload failed - ${result.error}`;
                        }
                    } catch (error) {
                        infoDiv.innerHTML = `<b>Preview Image:</b> Upload failed - ${error.message}`;
                    } finally {
                        uploadButton.disabled = false;
                        uploadButton.textContent = "Browse for preview image...";
                        fileInput.value = "";
                    }
                };
                
                reader.readAsDataURL(file);
            });
            
            // Update info when model changes
            const modelInput = node.inputs ? node.inputs.find(i => i.name === "model") : null;
            if (modelInput && modelInput.link) {
                infoDiv.innerHTML = "<b>Preview Image:</b> Connect model to see info";
            }
        }
        
        // Register a callback when nodes are added to the graph
        app.registerExtension({
            name: "ModelPresetPilot.UI",
            async nodeCreated(node) {
                if (node.comfyClass === "ModelPresetPilot") {
                    // Wait for widgets to be created
                    setTimeout(() => setupModelPresetPilotUI(node), 100);
                }
            }
        });
        """
        return js

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": ("STRING", {"default": "load", "choices": ["load", "save", "update"]}),
            },
            "optional": {
                # Identify the model
                "model": ("MODEL",),
                "ckpt_name": ("STRING", {"default": ""}),

                # Provide conditioning directly, or let us encode from CLIP + text
                "positive_cond": ("CONDITIONING",),
                "negative_cond": ("CONDITIONING",),
                "clip": ("CLIP",),
                "positive": ("STRING", {"default": "a photo of a subject"}),
                "negative": ("STRING", {"default": ""}),
                "vae": ("VAE",),

                # Preset fields (when saving/updating)
                "sampler_name": ("STRING", {"default": SAMPLER_CHOICES[0], "choices": SAMPLER_CHOICES}),
                "scheduler": ("STRING", {"default": SCHEDULER_CHOICES[0], "choices": SCHEDULER_CHOICES}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "clip_skip": ("INT", {"default": 0, "min": 0, "max": 12}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),

                # Preview generation control
                "generate_preview": ("BOOLEAN", {"default": False}),
                "overwrite_preview": ("BOOLEAN", {"default": False}),
                
                # Manual image loading
                "load_image": ("BOOLEAN", {"default": False}),
                "image": ("IMAGE", ),
                
                # File browser button (custom UI)
                "load_preview_from_file": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = (
        "STRING",   # model_id
        "STRING",   # sampler_name
        "STRING",   # scheduler
        "INT",      # steps
        "FLOAT",    # cfg
        "INT",      # clip_skip
        "INT",      # width
        "INT",      # height
        "INT",      # seed
        "IMAGE",    # preview
    )
    RETURN_NAMES = (
        "model_id", "sampler_name", "scheduler", "steps", "cfg",
        "clip_skip", "width", "height", "seed", "preview"
    )
    FUNCTION = "run"
    CATEGORY = "Custom/Model Presets"

    def _encode_if_needed(self, clip, positive, negative):
        if clip is None:
            raise ValueError("Preview generation requires either (positive_cond/negative_cond) or (clip + text).")
        enc = nodes.CLIPTextEncode()
        return enc.encode(clip, positive)[0], enc.encode(clip, negative)[0]

    def _generate_preview(self, model, vae, pos, neg, width, height, seed, steps, cfg, sampler, sched):
        latent = nodes.EmptyLatentImage().generate(width, height, batch_size=1)[0]
        out = nodes.KSampler().sample(
            model=model, seed=seed, steps=steps, cfg=cfg,
            sampler_name=sampler, scheduler=sched,
            positive=pos, negative=neg, latent_image=latent
        )[0]
        return nodes.VAEDecode().decode(vae, out)[0]

    def _empty_image(self):
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)

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
        
    def run(self, mode, model=None, ckpt_name="", positive_cond=None, negative_cond=None,
            clip=None, positive="a photo of a subject", negative="", vae=None,
            sampler_name=SAMPLER_CHOICES[0], scheduler=SCHEDULER_CHOICES[0],
            steps=28, cfg=5.5, clip_skip=0, width=1024, height=1024, seed=0,
            generate_preview=False, overwrite_preview=False, load_image=False, image=None,
            load_preview_from_file=False, unique_id=None, extra_pnginfo=None):

        model_id = _model_identifier(model, ckpt_name)
        preset = _load_preset(model_id)

        if mode == "load":
            sampler_name = preset.get("sampler_name", sampler_name)
            scheduler = preset.get("scheduler", scheduler)
            steps = preset.get("steps", steps)
            cfg = preset.get("cfg", cfg)
            clip_skip = preset.get("clip_skip", clip_skip)
            width = preset.get("width", width)
            height = preset.get("height", height)
            seed = preset.get("seed", seed)

        elif mode in ("save", "update"):
            if mode == "save":
                preset = {}
            for k, v in {
                "sampler_name": sampler_name, "scheduler": scheduler,
                "steps": steps, "cfg": cfg, "clip_skip": clip_skip,
                "width": width, "height": height, "seed": seed
            }.items():
                preset[k] = v
            _save_preset(model_id, preset)
        else:
            raise ValueError("Invalid mode. Choose among: load, save, update.")

        preview_file = _preview_path(model_id)
        preview_tensor = None
        
        # Handle manual image loading
        if load_image and image is not None and image.shape[0] > 0:
            # Save the provided image as the preview for this model
            self._save_image_as_preview(image, preview_file)
            preview_tensor = image
            
            # Get and display preview info
            preview_info = _get_preview_info(model_id)
            dimensions = preview_info.get("dimensions", "unknown")
            print(f"✅ Saved manually loaded image as preview for model: {model_id}")
            print(f"   - Dimensions: {dimensions}")
            print(f"   - Path: {preview_info['path']}")
        else:
            # Standard preview generation logic
            need_gen = generate_preview and (overwrite_preview or not os.path.exists(preview_file))
            can_gen = (model and vae) and (
                (positive_cond is not None and negative_cond is not None) or (clip and isinstance(positive, str))
            )

            if need_gen and can_gen:
                if positive_cond is None or negative_cond is None:
                    positive_cond, negative_cond = self._encode_if_needed(clip, positive, negative)
                img = self._generate_preview(model, vae, positive_cond, negative_cond,
                                            width, height, seed, steps, cfg, sampler_name, scheduler)
                self._save_image_as_preview(img, preview_file)
                preview_tensor = img

        # Load existing preview if we don't have one yet
        if preview_tensor is None and os.path.exists(preview_file):
            try:
                preview_tensor = _pil_to_image_tensor(Image.open(preview_file).convert("RGB"))
            except Exception:
                preview_tensor = None
                
        # Fallback to empty image if needed
        if preview_tensor is None:
            preview_tensor = self._empty_image()

        return (
            model_id, sampler_name, scheduler, steps, cfg,
            clip_skip, width, height, seed, preview_tensor
        )


# Add a custom UI component for the load image button
class ModelPresetPilotWidget:
    aux_id = "NewLouwa/ComfyUI-Model_preset_Pilot"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"node_id": ("STRING", {"multiline": False})}}
    
    RETURN_TYPES = ()
    FUNCTION = "load_image"
    OUTPUT_NODE = True
    CATEGORY = "Custom/Model Presets"

    def load_image(self, node_id):
        return {}

NODE_CLASS_MAPPINGS = {
    "ModelPresetPilot": ModelPresetPilot,
    "ModelPresetPilotWidget": ModelPresetPilotWidget
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelPresetPilot": "🧭 Model Preset Pilot",
    "ModelPresetPilotWidget": "🖼️ Load Model Preview Image"
}
