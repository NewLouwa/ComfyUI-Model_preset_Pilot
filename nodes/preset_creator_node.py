# Preset Creator Node
# Custom ComfyUI node for creating and saving model presets

import os
import json
import hashlib
import shutil
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import torch
import numpy as np
from PIL import Image

import comfy.utils
import nodes
import folder_paths


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
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
MODEL_DB_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "model_database.json")

os.makedirs(PRESET_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_DB_FILE), exist_ok=True)

# Setup logging
def setup_logger():
    logger = logging.getLogger('preset_creator')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler
    log_file = os.path.join(LOG_DIR, f"preset_creator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()


def _sanitize_filename(name: str) -> str:
    keep = "-_.() "
    safe = "".join(c for c in name if c.isalnum() or c in keep)
    safe = "_".join(safe.split())
    return safe[:128] or "model"

def _stem(s: str) -> str:
    base = os.path.basename(s)
    stem, _ = os.path.splitext(base)
    return stem

def _try_extract_ckpt_name_from_model(model):
    """Essaie plusieurs attributs fréquents pour extraire le nom du checkpoint"""
    logger.debug("=== EXTRACTING CKPT NAME FROM MODEL ===")
    
    # Try model attributes
    for attr in ("ckpt_path","ckptname","filename","orig_ckpt_name","model_name","name","path"):
        val = getattr(model, attr, None)
        logger.debug(f"model.{attr}: {val}")
        if val: 
            logger.info(f"Found checkpoint name in model.{attr}: {val}")
            return str(val)
    
    # Try inner model attributes
    inner = getattr(model, "model", None)
    if inner is not None:
        logger.debug(f"Inner model type: {type(inner)}")
        for attr in ("ckpt_path","ckptname","filename","model_name","name","path"):
            val = getattr(inner, attr, None)
            logger.debug(f"inner.{attr}: {val}")
            if val: 
                logger.info(f"Found checkpoint name in inner.{attr}: {val}")
                return str(val)
        
        # Try inner model __dict__
        d = getattr(inner, "__dict__", {})
        logger.debug(f"Inner model __dict__ keys: {list(d.keys())}")
        for k in ("ckpt_path","filename","model_name","name","path"):
            if d.get(k): 
                logger.info(f"Found checkpoint name in inner.__dict__.{k}: {d[k]}")
                return str(d[k])
    
    # Try to get from model's parent or other attributes
    if hasattr(model, 'parent') and model.parent is not None:
        logger.debug("Trying parent model...")
        for attr in ("ckpt_path","filename","model_name","name","path"):
            val = getattr(model.parent, attr, None)
            logger.debug(f"parent.{attr}: {val}")
            if val: 
                logger.info(f"Found checkpoint name in parent.{attr}: {val}")
                return str(val)
    
    logger.warning("No checkpoint name found in model attributes")
    return None

def _stable_model_hash(model):
    """Crée un hash stable basé sur le contenu du modèle"""
    try:
        inner = getattr(model, "model", model)
        if hasattr(inner, "state_dict"):
            sd = inner.state_dict()
            for _, v in sd.items():
                if isinstance(v, torch.Tensor):
                    b = v.detach().cpu().numpy().tobytes()
                    return hashlib.sha256(b[:1024*1024]).hexdigest()[:12]
    except Exception:
        pass
    return hashlib.sha256(repr(model).encode()).hexdigest()[:12]

def _best_ckpt_choice(detected: Optional[str], choices):
    """Retourne un nom de checkpoint présent dans choices si on peut matcher.
       Matching tolérant: exact, basename, puis stem (sans extension), case-insensitive."""
    if not detected:
        return None
    det = os.path.basename(str(detected))
    det_stem = _stem(det).lower()

    # 1) exact
    for c in choices:
        if c == det:
            return c
    # 2) basename-insensitive
    for c in choices:
        if c.lower() == det.lower():
            return c
    # 3) stem match
    for c in choices:
        if _stem(c).lower() == det_stem:
            return c
    # 4) stem "starts with" (utile si suffixes .safetensors / hashes)
    for c in choices:
        if _stem(c).lower().startswith(det_stem):
            return c
    return None

def _load_model_database():
    """Charge la base de données des modèles"""
    if os.path.exists(MODEL_DB_FILE):
        try:
            with open(MODEL_DB_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load model database: {e}")
    return {"models": {}, "last_updated": None}

def _save_model_database(db):
    """Sauvegarde la base de données des modèles"""
    db["last_updated"] = datetime.now().isoformat()
    try:
        with open(MODEL_DB_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Could not save model database: {e}")

def _update_model_database(model_id: str, checkpoint_name: str = None, model_info: dict = None):
    """Met à jour la base de données des modèles"""
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
        # Mettre à jour les informations existantes
        if checkpoint_name and not db["models"][model_id]["checkpoint_name"]:
            db["models"][model_id]["checkpoint_name"] = checkpoint_name
        db["models"][model_id]["last_seen"] = datetime.now().isoformat()
        db["models"][model_id]["usage_count"] = db["models"][model_id].get("usage_count", 0) + 1
    
    _save_model_database(db)
    logger.info(f"Updated model database: {model_id} -> {checkpoint_name}")

def _get_model_name_from_database(model_id: str) -> str:
    """Récupère le nom du modèle depuis la base de données"""
    db = _load_model_database()
    if model_id in db["models"]:
        return db["models"][model_id].get("checkpoint_name", model_id)
    return model_id

def _model_identifier_from_ckpt(ckpt_name: Optional[str], model) -> str:
    """Crée un identifiant de modèle stable"""
    # Essayer d'extraire le nom depuis le modèle
    name = _try_extract_ckpt_name_from_model(model)
    if name:
        model_id = _sanitize_filename(_stem(name))
        _update_model_database(model_id, name)
        return model_id
    
    # Utiliser un hash stable basé sur le contenu du modèle
    model_id = f"model_{_stable_model_hash(model)}"
    _update_model_database(model_id, None, {"hash_based": True})
    return model_id


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
    logger.info("=== MODEL IDENTIFIER DEBUG ===")
    
    if model is not None:
        # Debug: Log model attributes
        logger.debug(f"Model type: {type(model)}")
        
        # Try to get the model name from ComfyUI's model loading system
        try:
            # Check if we can get the model name from the model's internal structure
            if hasattr(model, 'model') and hasattr(model.model, '__class__'):
                model_class_name = model.model.__class__.__name__
                logger.debug(f"Model class name: {model_class_name}")
                
                # Try to get the model name from the model's config or state
                if hasattr(model.model, 'config') and hasattr(model.model.config, 'name'):
                    name = model.model.config.name
                    logger.info(f"Found model name in config: {name}")
                elif hasattr(model.model, 'name'):
                    name = model.model.name
                    logger.info(f"Found model name in model.name: {name}")
                else:
                    name = None
        except Exception as e:
            logger.debug(f"Could not get model name from ComfyUI system: {e}")
            name = None
        
        # Try to get model name from ComfyUI's model loading system
        try:
            # Check if we can access the model's file path through ComfyUI's system
            if hasattr(model, 'model') and hasattr(model.model, 'config'):
                # Try to get the model name from the config
                config = model.model.config
                if hasattr(config, 'name'):
                    name = config.name
                    logger.info(f"Found model name in config.name: {name}")
                elif hasattr(config, 'model_name'):
                    name = config.model_name
                    logger.info(f"Found model name in config.model_name: {name}")
                elif hasattr(config, 'ckpt_path'):
                    name = config.ckpt_path
                    logger.info(f"Found model name in config.ckpt_path: {name}")
        except Exception as e:
            logger.debug(f"Could not get model name from config: {e}")
        
        # Try to get model name from ComfyUI's model loading system
        try:
            # Check if we can get the model name from the model's state dict
            if hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
                # Try to get the model name from the state dict
                state_dict = model.model.state_dict()
                if hasattr(state_dict, 'keys'):
                    # Look for common model name keys in the state dict
                    for key in state_dict.keys():
                        if 'model_name' in key.lower() or 'name' in key.lower():
                            if isinstance(state_dict[key], str):
                                name = state_dict[key]
                                logger.info(f"Found model name in state_dict.{key}: {name}")
                                break
        except Exception as e:
            logger.debug(f"Could not get model name from state_dict: {e}")
        
        # Try model.model_name first
        if not name:
            name = getattr(model, "model_name", None)
            logger.debug(f"model.model_name: {name}")
        
        if not name:
            # Try inner model attributes
            inner = getattr(model, "model", None)
            if inner:
                name = getattr(inner, "model_name", None) or getattr(inner, "ckpt_path", None)
                logger.debug(f"inner.model_name/ckpt_path: {name}")
        
        # If still no name, try other common attributes
        if not name:
            # Try common model attributes that might contain filename
            for attr in ["filename", "file_name", "path", "ckpt_path", "name", "config_path"]:
                if hasattr(model, attr):
                    attr_value = getattr(model, attr)
                    logger.debug(f"model.{attr}: {attr_value}")
                    # Skip methods and None values, only use strings
                    if attr_value and str(attr_value) != "None" and not callable(attr_value) and isinstance(attr_value, str):
                        name = attr_value
                        logger.info(f"Found model name in model.{attr}: {name}")
                        break
                # Also check inner model
                if inner and hasattr(inner, attr):
                    attr_value = getattr(inner, attr)
                    logger.debug(f"inner.{attr}: {attr_value}")
                    if attr_value and str(attr_value) != "None" and not callable(attr_value) and isinstance(attr_value, str):
                        name = attr_value
                        logger.info(f"Found model name in inner.{attr}: {name}")
                        break
        
        # If we found a name, extract just the filename without extension
        if name:
            name_str = str(name)
            logger.info(f"Found name: {name_str}")
            # Extract filename from path if it's a full path
            if "/" in name_str or "\\" in name_str:
                name_str = os.path.basename(name_str)
                logger.debug(f"Extracted filename from path: {name_str}")
            # Remove extension
            name_str = os.path.splitext(name_str)[0]
            result = _sanitize_filename(name_str)
            logger.info(f"Final model identifier: {result}")
            return result
    
    if ckpt_name:
        result = _sanitize_filename(str(ckpt_name))
        logger.info(f"Using ckpt_name: {result}")
        return result
    
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
            logger.info(f"Using stable model hash: {stable_hash}")
            return f"model_{stable_hash}"
    except Exception as e:
        logger.debug(f"Could not create stable hash: {e}")
    
    fallback = f"unknown_{hashlib.sha256(repr(model).encode()).hexdigest()[:10]}"
    logger.warning(f"Using fallback identifier: {fallback}")
    return fallback


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
        # No custom JavaScript needed - using ComfyUI's native boolean widget
        return ""

    @classmethod
    def INPUT_TYPES(cls):
        # Get available checkpoints for model selection
        try:
            checkpoint_choices = folder_paths.get_filename_list("checkpoints")
            if not checkpoint_choices:
                checkpoint_choices = ["<no checkpoints found>"]
        except Exception:
            checkpoint_choices = ["<no checkpoints found>"]
        
        return {
            "required": {
                "preset_name": ("STRING", {"default": ""}),
                "save_preset": ("BOOLEAN", {"default": False, "label": "💾 Save Preset"}),
                "model_name": (checkpoint_choices, {"default": checkpoint_choices[0] if checkpoint_choices else "<none>"}),
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
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = (
        "STRING",   # status message
    )
    RETURN_NAMES = (
        "status"
    )
    FUNCTION = "run"
    CATEGORY = "🤖 Model Preset Pilot"

        
    def run(self, preset_name="", save_preset=False, model_name=None, sampler_name=None, scheduler=None, steps=None, cfg=None, clip_skip=None, width=None, height=None, seed=None, 
            unique_id=None, extra_pnginfo=None):
        
        logger.info("=== PRESET CREATOR RUN ===")
        logger.info(f"Save preset: {save_preset}")
        logger.info(f"Preset name: {preset_name}")
        logger.info(f"Model name: {model_name}")
        logger.info(f"Parameters: sampler={sampler_name}, scheduler={scheduler}, steps={steps}, cfg={cfg}")
        
        # Utiliser le nom du modèle sélectionné
        if model_name and model_name != "<no checkpoints found>" and model_name != "<none>":
            # Utiliser le nom sélectionné par l'utilisateur
            model_id = _sanitize_filename(_stem(model_name))
            logger.info(f"Using selected model name: {model_name} -> {model_id}")
            
            # Mettre à jour la base de données avec le nom lisible
            _update_model_database(model_id, model_name, {"user_selected": True})
        else:
            logger.error("No model name selected!")
            status = "❌ Error: Please select a model name"
            return (status,)
        
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
            
            logger.info(f"Creating preset for model '{model_id}' with name '{preset_name}'")
            logger.debug(f"Preset data: {preset_data}")
            
            try:
                # Create preset using storage manager
                preset_id = create_preset(model_id, preset_data, preset_name)
                logger.info(f"Successfully created preset: {preset_id}")
                
                # No preview image handling needed
                
                # Create status message
                status = f"✅ Preset saved!\n📝 Name: {preset_id}\n📁 Location: {model_id}/{preset_id}\n⚙️ Settings: {sampler_name}, {scheduler}, {steps} steps"
                logger.info(f"Status: {status}")
            except Exception as e:
                logger.error(f"Failed to create preset: {str(e)}", exc_info=True)
                status = f"❌ Failed to save preset: {str(e)}"
        else:
            # Just show current settings without saving
            logger.info("Save preset not enabled")
            status = f"📋 Current Settings:\n⚙️ {sampler_name}, {scheduler}, {steps} steps\n💾 Click 'Save Preset' to save these settings"
        
        return (status,)
