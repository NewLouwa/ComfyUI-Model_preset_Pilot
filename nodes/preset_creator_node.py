# Preset Creator Node
# Custom ComfyUI node for creating and saving model presets

import os
import json
import hashlib
import shutil
from typing import Any, Dict, Optional

import torch
from PIL import Image

import comfy.utils
import nodes
import folder_paths

# Try to read available samplers/schedulers from Comfy's KSampler
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
os.makedirs(PRESET_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)


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


def _save_preset(model_id: str, data: Dict[str, Any]) -> None:
    with open(_preset_path(model_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _image_tensor_to_pil(t: torch.Tensor) -> Image.Image:
    # t: [B, H, W, C], 0..1 float
    t = t[0].clamp(0, 1)
    arr = (t.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


class PresetCreator:
    """
    Preset Creator - creates and saves model presets with current settings
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
        # Add custom JavaScript for the Preset Creator UI
        js = """
        // Add custom JavaScript for the Preset Creator node
        function setupPresetCreatorUI(node) {
            // Create save button
            const saveButton = document.createElement("button");
            saveButton.textContent = "💾 Save Preset";
            saveButton.style.margin = "10px 0";
            saveButton.style.padding = "8px 16px";
            saveButton.style.backgroundColor = "#2a5a2a";
            saveButton.style.color = "white";
            saveButton.style.border = "none";
            saveButton.style.borderRadius = "6px";
            saveButton.style.cursor = "pointer";
            saveButton.style.fontSize = "14px";
            saveButton.style.fontWeight = "bold";
            saveButton.style.width = "100%";
            
            // Add hover effect
            saveButton.addEventListener("mouseover", () => {
                saveButton.style.backgroundColor = "#3a6a3a";
            });
            saveButton.addEventListener("mouseout", () => {
                saveButton.style.backgroundColor = "#2a5a2a";
            });
            
            // Create status display
            const statusDisplay = document.createElement("div");
            statusDisplay.className = "preset-creator-status";
            statusDisplay.style.margin = "10px 0";
            statusDisplay.style.padding = "8px";
            statusDisplay.style.backgroundColor = "rgba(0,0,0,0.3)";
            statusDisplay.style.borderRadius = "4px";
            statusDisplay.style.fontSize = "12px";
            statusDisplay.style.fontFamily = "monospace";
            statusDisplay.style.color = "#ccc";
            statusDisplay.style.border = "1px solid #444";
            statusDisplay.innerHTML = "Ready to save preset";
            
            // Add elements to the node
            const nodeContainer = node.widgets[0].options.el.parentElement.parentElement;
            nodeContainer.appendChild(saveButton);
            nodeContainer.appendChild(statusDisplay);
            
            // Handle save button click
            saveButton.addEventListener("click", () => {
                statusDisplay.innerHTML = "💾 Saving preset...";
                statusDisplay.style.color = "#ffa500";
                
                // Set the save_preset widget to true
                const saveWidget = node.widgets.find(w => w.name === "save_preset");
                if (saveWidget) {
                    saveWidget.value = true;
                }
                
                // Trigger node execution
                node.setDirtyCanvas(true);
            });
        }
        
        // Register a callback when nodes are added to the graph
        app.registerExtension({
            name: "PresetCreator.UI",
            async nodeCreated(node) {
                if (node.comfyClass === "PresetCreator") {
                    // Wait for widgets to be created
                    setTimeout(() => setupPresetCreatorUI(node), 100);
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
                "sampler_name": ("STRING", {"default": SAMPLER_CHOICES[0], "choices": SAMPLER_CHOICES}),
                "scheduler": ("STRING", {"default": SCHEDULER_CHOICES[0], "choices": SCHEDULER_CHOICES}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "clip_skip": ("INT", {"default": 0, "min": 0, "max": 12}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "preset_name": ("STRING", {"default": ""}),
            },
            "widgets": {
                "save_preset": ("BUTTON", {"label": "💾 Save Preset"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = (
        "MODEL",    # model (pass through)
        "STRING",   # status message
    )
    RETURN_NAMES = (
        "model", "status"
    )
    FUNCTION = "run"
    CATEGORY = "🤖 Model Preset Pilot"

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
        
    def run(self, model, sampler_name, scheduler, steps, cfg, clip_skip, width, height, seed, 
            image=None, preset_name="", save_preset=False, unique_id=None, extra_pnginfo=None):
        
        # Get model identifier
        model_id = _model_identifier(model, None)
        
        if save_preset:
            # Import storage manager functions
            from .storage_manager import create_preset, save_preview_image
            
            # Create preset data
            preset_data = {
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "steps": steps,
                "cfg": cfg,
                "clip_skip": clip_skip,
                "width": width,
                "height": height,
                "seed": seed,
                "description": f"Preset created for {model_id}",
                "tags": ["user_created"]
            }
            
            # Create preset using storage manager
            preset_id = create_preset(model_id, preset_data, preset_name)
            
            # Save preview image if provided
            if image is not None and image.shape[0] > 0:
                save_preview_image(model_id, preset_id, image)
            
            # Create status message
            status = f"✅ Preset saved!\n📝 Name: {preset_id}\n📁 Location: {model_id}/{preset_id}\n⚙️ Settings: {sampler_name}, {scheduler}, {steps} steps"
        else:
            # Just show current settings without saving
            status = f"📋 Current Settings:\n⚙️ {sampler_name}, {scheduler}, {steps} steps\n💾 Click 'Save Preset' to save these settings"
        
        return (model, status)
