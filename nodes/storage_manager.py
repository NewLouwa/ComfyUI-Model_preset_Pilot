# Storage Manager for Model Presets
# Handles all preset storage operations with a flexible structure

import os
import json
import hashlib
import shutil
import logging
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

# Setup logging
LOG_DIR = os.path.join(NODE_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger():
    logger = logging.getLogger('storage_manager')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler
    log_file = os.path.join(LOG_DIR, f"storage_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()


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


def _get_preset_directory(model_id: str, preset_id: str) -> str:
    """Get the directory path for a specific preset"""
    return os.path.join(_get_model_directory(model_id), preset_id)


def _get_preset_file(model_id: str, preset_id: str) -> str:
    """Get the preset file path for a specific preset"""
    return os.path.join(_get_preset_directory(model_id, preset_id), "preset.json")


def _get_preset_preview_file(model_id: str, preset_id: str) -> str:
    """Get the preview file path for a specific preset"""
    return os.path.join(_get_preset_directory(model_id, preset_id), "preview.png")


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
    logger.info(f"=== CREATE PRESET ===")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Preset name: {preset_name}")
    logger.info(f"Preset data: {preset_data}")
    
    # Generate preset ID
    preset_id = preset_name or f"preset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    preset_id = _sanitize_filename(preset_id)
    logger.info(f"Generated preset ID: {preset_id}")
    
    # Create preset directory
    preset_dir = _get_preset_directory(model_id, preset_id)
    logger.info(f"Preset directory: {preset_dir}")
    os.makedirs(preset_dir, exist_ok=True)
    logger.info(f"Created preset directory: {preset_dir}")
    
    # Create preset entry
    preset_entry = {
        "id": preset_id,
        "name": preset_name or f"Preset {preset_id}",
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
        "tags": preset_data.get("tags", [])
    }
    
    logger.debug(f"Preset entry: {preset_entry}")
    
    # Save individual preset file
    preset_file = _get_preset_file(model_id, preset_id)
    logger.info(f"Saving preset file: {preset_file}")
    with open(preset_file, "w", encoding="utf-8") as f:
        json.dump(preset_entry, f, ensure_ascii=False, indent=2)
    logger.info(f"Preset file saved successfully")
    
    # Update model metadata
    metadata = _load_model_metadata(model_id)
    model_dir = _get_model_directory(model_id)
    logger.info(f"Model directory: {model_dir}")
    
    # Count existing presets
    preset_count = 0
    if os.path.exists(model_dir):
        preset_count = len([d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))])
    
    metadata["preset_count"] = preset_count
    metadata["last_used"] = datetime.now().isoformat()
    logger.info(f"Updated metadata - preset count: {preset_count}")
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
        logger.info(f"Added model to index: {model_id}")
    _save_index(index_data)
    logger.info(f"Index updated successfully")
    
    return preset_id


def get_preset(model_id: str, preset_id: str = None) -> Dict[str, Any]:
    """Get a specific preset for a model"""
    if preset_id:
        preset_file = _get_preset_file(model_id, preset_id)
        if os.path.exists(preset_file):
            try:
                with open(preset_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load preset {preset_id}: {e}")
        return {}
    else:
        # Return first available preset
        model_dir = _get_model_directory(model_id)
        if os.path.exists(model_dir):
            for item in os.listdir(model_dir):
                if os.path.isdir(os.path.join(model_dir, item)) and item != "metadata.json":
                    preset_file = _get_preset_file(model_id, item)
                    if os.path.exists(preset_file):
                        try:
                            with open(preset_file, "r", encoding="utf-8") as f:
                                return json.load(f)
                        except Exception as e:
                            print(f"Warning: Could not load preset {item}: {e}")
        return {}


def get_all_presets(model_id: str) -> Dict[str, Any]:
    """Get all presets for a model"""
    presets = {}
    model_dir = _get_model_directory(model_id)
    
    if os.path.exists(model_dir):
        for item in os.listdir(model_dir):
            if os.path.isdir(os.path.join(model_dir, item)) and item != "metadata.json":
                preset_file = _get_preset_file(model_id, item)
                if os.path.exists(preset_file):
                    try:
                        with open(preset_file, "r", encoding="utf-8") as f:
                            presets[item] = json.load(f)
                    except Exception as e:
                        print(f"Warning: Could not load preset {item}: {e}")
    
    return {"presets": presets}


def save_preview_image(model_id: str, preset_id: str, image_tensor: torch.Tensor) -> str:
    """Save a preview image for a preset"""
    preview_file = _get_preset_preview_file(model_id, preset_id)
    
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
    preview_file = _get_preset_preview_file(model_id, preset_id)
    
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
    preset_dir = _get_preset_directory(model_id, preset_id)
    
    if os.path.exists(preset_dir):
        # Remove entire preset directory
        shutil.rmtree(preset_dir)
        
        # Update model metadata
        metadata = _load_model_metadata(model_id)
        metadata["preset_count"] = len([d for d in os.listdir(_get_model_directory(model_id)) if os.path.isdir(os.path.join(_get_model_directory(model_id), d))])
        _save_model_metadata(model_id, metadata)
        
        return True
    return False


# Model database functions (moved from preset_creator_node.py)
MODEL_DB_FILE = os.path.join(NODE_ROOT, "data", "model_database.json")

def _load_model_database():
    """Load the model database"""
    if os.path.exists(MODEL_DB_FILE):
        try:
            with open(MODEL_DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load model database: {e}")
    return {"models": {}, "last_updated": None}

def _save_model_database(db):
    """Save the model database"""
    db["last_updated"] = datetime.now().isoformat()
    try:
        with open(MODEL_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Could not save model database: {e}")

def _update_model_database(model_id: str, checkpoint_name: str = None, model_info: dict = None):
    """Update the model database"""
    db = _load_model_database()
    
    if model_id not in db["models"]:
        db["models"][model_id] = {
            "id": model_id,
            "checkpoint_name": checkpoint_name,
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "usage_count": 0,
            "model_info": model_info or {}
        }
    else:
        # Update existing information
        if checkpoint_name and not db["models"][model_id]["checkpoint_name"]:
            db["models"][model_id]["checkpoint_name"] = checkpoint_name
        db["models"][model_id]["last_seen"] = datetime.now().isoformat()
        db["models"][model_id]["usage_count"] = db["models"][model_id].get("usage_count", 0) + 1
    
    _save_model_database(db)
    logger.info(f"Updated model database: {model_id} -> {checkpoint_name}")

def _get_model_name_from_database(model_id: str) -> str:
    """Get the model name from the database"""
    db = _load_model_database()
    if model_id in db["models"]:
        return db["models"][model_id].get("checkpoint_name", model_id)
    return model_id
