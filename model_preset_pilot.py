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
from typing import Any, Dict, Optional

import torch
from PIL import Image

import comfy.utils
import nodes

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
            },
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

    def run(self, mode, model=None, ckpt_name="", positive_cond=None, negative_cond=None,
            clip=None, positive="a photo of a subject", negative="", vae=None,
            sampler_name=SAMPLER_CHOICES[0], scheduler=SCHEDULER_CHOICES[0],
            steps=28, cfg=5.5, clip_skip=0, width=1024, height=1024, seed=0,
            generate_preview=False, overwrite_preview=False):

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
        need_gen = generate_preview and (overwrite_preview or not os.path.exists(preview_file))
        can_gen = (model and vae) and (
            (positive_cond is not None and negative_cond is not None) or (clip and isinstance(positive, str))
        )

        if need_gen and can_gen:
            if positive_cond is None or negative_cond is None:
                positive_cond, negative_cond = self._encode_if_needed(clip, positive, negative)
            img = self._generate_preview(model, vae, positive_cond, negative_cond,
                                         width, height, seed, steps, cfg, sampler_name, scheduler)
            _image_tensor_to_pil(img).save(preview_file)
            preview_tensor = img

        if preview_tensor is None and os.path.exists(preview_file):
            try:
                preview_tensor = _pil_to_image_tensor(Image.open(preview_file).convert("RGB"))
            except Exception:
                preview_tensor = None
        if preview_tensor is None:
            preview_tensor = self._empty_image()

        return (
            model_id, sampler_name, scheduler, steps, cfg,
            clip_skip, width, height, seed, preview_tensor
        )


NODE_CLASS_MAPPINGS = {"ModelPresetPilot": ModelPresetPilot}
NODE_DISPLAY_NAME_MAPPINGS = {"ModelPresetPilot": "🧭 Model Preset Pilot"}
