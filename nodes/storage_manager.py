# Storage Manager for Model Presets
# Handles all preset storage operations with a flexible structure

import os
import json
import hashlib
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# Storage configuration - Keep everything within the node module
NODE_ROOT = os.path.dirname(os.path.dirname(__file__))  # Go up to ComfyUI-Model_preset_Pilot
BASE_PRESET_DIR = os.path.join(NODE_ROOT, "data", "presets")
MODELS_DIR = os.path.join(BASE_PRESET_DIR, "models")
INDEX_FILE = os.path.join(BASE_PRESET_DIR, "index.json")

# Ensure directories exist
os.makedirs(BASE_PRESET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    """Sanitize filename for safe storage"""
    keep = "-_.() "
    safe = "".join(c for c in name if c.isalnum() or c in keep)
    safe = "_".join(safe.split())
    return safe[:128] or "model"


def _model_identifier(model: Optional[Any], ckpt_name: Optional[str]) -> str:
    """Create a stable identifier for the model"""
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


def _get_model_directory(model_id: str) -> str:
    """Get the directory path for a specific model"""
    return os.path.join(MODELS_DIR, model_id)


def _get_model_presets_file(model_id: str) -> str:
    """Get the presets file path for a specific model"""
    return os.path.join(_get_model_directory(model_id), "presets.json")


def _get_model_previews_dir(model_id: str) -> str:
    """Get the previews directory for a specific model"""
    return os.path.join(_get_model_directory(model_id), "previews")


def _get_model_metadata_file(model_id: str) -> str:
    """Get the metadata file path for a specific model"""
    return os.path.join(_get_model_directory(model_id), "metadata.json")


def _load_index() -> Dict[str, Any]:
    """Load the main index file"""
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load index file: {e}")
    return {"models": {}, "last_updated": None}


def _save_index(index_data: Dict[str, Any]) -> None:
    """Save the main index file"""
    index_data["last_updated"] = datetime.now().isoformat()
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)


def _load_model_presets(model_id: str) -> Dict[str, Any]:
    """Load all presets for a specific model"""
    presets_file = _get_model_presets_file(model_id)
    if os.path.exists(presets_file):
        try:
            with open(presets_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load presets for {model_id}: {e}")
    return {"presets": {}, "default_preset": None}


def _save_model_presets(model_id: str, presets_data: Dict[str, Any]) -> None:
    """Save all presets for a specific model"""
    model_dir = _get_model_directory(model_id)
    os.makedirs(model_dir, exist_ok=True)
    
    presets_file = _get_model_presets_file(model_id)
    with open(presets_file, "w", encoding="utf-8") as f:
        json.dump(presets_data, f, ensure_ascii=False, indent=2)


def _load_model_metadata(model_id: str) -> Dict[str, Any]:
    """Load metadata for a specific model"""
    metadata_file = _get_model_metadata_file(model_id)
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata for {model_id}: {e}")
    return {
        "model_id": model_id,
        "model_name": "Unknown Model",
        "created_at": datetime.now().isoformat(),
        "last_used": None,
        "preset_count": 0,
        "tags": []
    }


def _save_model_metadata(model_id: str, metadata: Dict[str, Any]) -> None:
    """Save metadata for a specific model"""
    model_dir = _get_model_directory(model_id)
    os.makedirs(model_dir, exist_ok=True)
    
    metadata_file = _get_model_metadata_file(model_id)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def create_preset(model_id: str, preset_data: Dict[str, Any], preset_name: str = None) -> str:
    """Create a new preset for a model"""
    # Load existing presets
    presets_data = _load_model_presets(model_id)
    
    # Generate preset ID
    preset_id = preset_name or f"preset_{len(presets_data.get('presets', {})) + 1}"
    preset_id = _sanitize_filename(preset_id)
    
    # Create preset entry
    preset_entry = {
        "id": preset_id,
        "name": preset_name or f"Preset {len(presets_data.get('presets', {})) + 1}",
        "description": preset_data.get("description", ""),
        "sampler_name": preset_data.get("sampler_name", "none"),
        "scheduler": preset_data.get("scheduler", "none"),
        "steps": preset_data.get("steps", 0),
        "cfg": preset_data.get("cfg", 0.0),
        "clip_skip": preset_data.get("clip_skip", 0),
        "width": preset_data.get("width", 0),
        "height": preset_data.get("height", 0),
        "seed": preset_data.get("seed", 0),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "preview_image": f"{preset_id}.png",
        "tags": preset_data.get("tags", [])
    }
    
    # Add to presets
    if "presets" not in presets_data:
        presets_data["presets"] = {}
    presets_data["presets"][preset_id] = preset_entry
    
    # Set as default if it's the first preset
    if not presets_data.get("default_preset"):
        presets_data["default_preset"] = preset_id
    
    # Save presets
    _save_model_presets(model_id, presets_data)
    
    # Update model metadata
    metadata = _load_model_metadata(model_id)
    metadata["preset_count"] = len(presets_data["presets"])
    metadata["last_used"] = datetime.now().isoformat()
    _save_model_metadata(model_id, metadata)
    
    # Update main index
    index_data = _load_index()
    if model_id not in index_data["models"]:
        index_data["models"][model_id] = {
            "model_name": metadata.get("model_name", "Unknown Model"),
            "created_at": metadata.get("created_at"),
            "last_used": metadata.get("last_used"),
            "preset_count": metadata.get("preset_count", 0)
        }
    _save_index(index_data)
    
    return preset_id


def get_preset(model_id: str, preset_id: str = None) -> Dict[str, Any]:
    """Get a specific preset for a model"""
    presets_data = _load_model_presets(model_id)
    
    if preset_id:
        return presets_data.get("presets", {}).get(preset_id, {})
    else:
        # Return default preset
        default_id = presets_data.get("default_preset")
        if default_id:
            return presets_data.get("presets", {}).get(default_id, {})
        # Return first available preset
        presets = presets_data.get("presets", {})
        if presets:
            return list(presets.values())[0]
        return {}


def get_all_presets(model_id: str) -> Dict[str, Any]:
    """Get all presets for a model"""
    return _load_model_presets(model_id)


def save_preview_image(model_id: str, preset_id: str, image_tensor: torch.Tensor) -> str:
    """Save a preview image for a preset"""
    previews_dir = _get_model_previews_dir(model_id)
    os.makedirs(previews_dir, exist_ok=True)
    
    preview_file = os.path.join(previews_dir, f"{preset_id}.png")
    
    # Convert tensor to PIL and save
    if image_tensor is not None and image_tensor.shape[0] > 0:
        t = image_tensor[0].clamp(0, 1)
        arr = (t.cpu().numpy() * 255).astype("uint8")
        img = Image.fromarray(arr)
        img.save(preview_file)
        return preview_file
    return ""


def load_preview_image(model_id: str, preset_id: str) -> Optional[torch.Tensor]:
    """Load a preview image for a preset"""
    preview_file = os.path.join(_get_model_previews_dir(model_id), f"{preset_id}.png")
    
    if os.path.exists(preview_file):
        try:
            img = Image.open(preview_file).convert("RGB")
            arr = torch.from_numpy(img).float() / 255.0
            return arr.unsqueeze(0)
        except Exception as e:
            print(f"Warning: Could not load preview image: {e}")
    return None


def list_models() -> List[Dict[str, Any]]:
    """List all models with their metadata"""
    index_data = _load_index()
    return list(index_data.get("models", {}).values())


def delete_preset(model_id: str, preset_id: str) -> bool:
    """Delete a specific preset"""
    presets_data = _load_model_presets(model_id)
    
    if preset_id in presets_data.get("presets", {}):
        # Remove preset
        del presets_data["presets"][preset_id]
        
        # Update default if this was the default
        if presets_data.get("default_preset") == preset_id:
            presets = presets_data.get("presets", {})
            presets_data["default_preset"] = list(presets.keys())[0] if presets else None
        
        # Save updated presets
        _save_model_presets(model_id, presets_data)
        
        # Update metadata
        metadata = _load_model_metadata(model_id)
        metadata["preset_count"] = len(presets_data.get("presets", {}))
        _save_model_metadata(model_id, metadata)
        
        # Delete preview image
        preview_file = os.path.join(_get_model_previews_dir(model_id), f"{preset_id}.png")
        if os.path.exists(preview_file):
            os.remove(preview_file)
        
        return True
    return False
