# model_preset_loader_internal.py
# Internal preview-only loader for Model Preset Pilot
# - No IMAGE output: preview is rendered inside the node card (canvas)
# - Dynamic resizing (contain fit, high-DPI aware)
# - HTTP GET route to serve preset preview PNGs
# - Scans preset directories to build dropdown "ModelDisplay/PresetID"

import os
import io
import json
import hashlib
from typing import Dict, Optional

import torch
import numpy as np
from PIL import Image

import nodes
import folder_paths
import comfy.utils

# Comfy server (Starlette) bits
import server
from server import PromptServer
from starlette.responses import Response

# ------------------------------
# Paths & data setup
# ------------------------------
DATA_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DEFAULTS_DIR = os.path.join(DATA_DIR, "defaults")
ASSETS_DIR   = os.path.join(DATA_DIR, "assets")
PRESET_DIR   = os.path.join(DATA_DIR, "presets")

os.makedirs(DEFAULTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR,   exist_ok=True)
os.makedirs(PRESET_DIR,   exist_ok=True)

# ------------------------------
# Helpers (PIL <-> tensor, defaults)
# ------------------------------
def _pil_to_image_tensor(pil: Image.Image) -> torch.Tensor:
    """PIL -> Comfy tensor [1,H,W,C], float32 in [0,1]"""
    img = pil.convert("RGB")
    arr = np.array(img)  # (H,W,C) uint8
    t = torch.from_numpy(arr).float() / 255.0
    return t.unsqueeze(0)

def _image_tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Comfy tensor [1,H,W,C] -> PIL"""
    t = t[0].clamp(0, 1)
    arr = (t.cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)

def _default_preview_tensor() -> torch.Tensor:
    # Essaye un visuel par défaut dans data/defaults
    for name in ("NothingHere_Robot.png",
                 "NothingHere_Robot2.png",
                 "NothingHere_Robot3.png",
                 "NothingHere_Robot4.png"):
        p = os.path.join(DEFAULTS_DIR, name)
        if os.path.exists(p):
            try:
                return _pil_to_image_tensor(Image.open(p))
            except Exception:
                pass
    # Sinon, placeholder gris 512
    pil = Image.new("RGB", (512, 512), "gray")
    return _pil_to_image_tensor(pil)

# ------------------------------
# Storage Manager (contract minimal)
# ------------------------------
# On réutilise ton storage_manager pour localiser previews & DB.
try:
    from .storage_manager import _load_model_database, _get_preset_preview_file  # type: ignore
except Exception:
    # Fallbacks neutres si storage_manager absent (node toujours fonctionnel en "none")
    def _load_model_database() -> Dict:
        return {"models": {}}
    def _get_preset_preview_file(model_id: str, preset_id: str) -> str:
        return os.path.join(PRESET_DIR, "models", model_id, preset_id, "preview.png")

# ------------------------------
# GET route: /model_preset_pilot/preview?preset_name=Display/Preset
# Renvoie un PNG (preset preview ou fallback)
# ------------------------------
def _png_response_from_tensor(t: torch.Tensor) -> Response:
    pil = _image_tensor_to_pil(t)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")

def api_preview_from_choice(preset_name: Optional[str] = None):
    try:
        # format attendu: "DisplayName/PresetID"
        if not preset_name or "/" not in preset_name:
            return _png_response_from_tensor(_default_preview_tensor())

        display_name, preset_id = preset_name.split("/", 1)

        # Résoudre model_id via la DB (checkpoint_name stem == display_name)
        db = _load_model_database()
        models = db.get("models", {})
        actual_model_id = None

        for mid, entry in models.items():
            ck = entry.get("checkpoint_name", "")
            if not ck:
                continue
            stem = os.path.splitext(os.path.basename(ck))[0]
            if stem.lower() == display_name.lower():
                actual_model_id = mid
                break

        # Fallback: match partiel sur dossiers
        if actual_model_id is None:
            models_root = os.path.join(PRESET_DIR, "models")
            if os.path.isdir(models_root):
                for mid in os.listdir(models_root):
                    if display_name.lower() in mid.lower() or mid.lower() in display_name.lower():
                        actual_model_id = mid
                        break

        if actual_model_id:
            path = _get_preset_preview_file(actual_model_id, preset_id)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data = f.read()
                return Response(data, media_type="image/png")

        # Fallback image
        return _png_response_from_tensor(_default_preview_tensor())

    except Exception:
        return _png_response_from_tensor(_default_preview_tensor())

# Enregistrer la route GET (compatible anciennes/nouvelles versions)
try:
    if hasattr(PromptServer.instance, "add_api_route"):
        PromptServer.instance.add_api_route("/model_preset_pilot/preview", api_preview_from_choice, methods=["GET"])
    else:
        PromptServer.instance.app.add_route("/model_preset_pilot/preview", api_preview_from_choice, methods=["GET"])
except Exception as e:
    print(f"[ModelPresetPilot] Warning: could not register GET route: {e}")

# ------------------------------
# Utilitaires: scan presets -> dropdown
# ------------------------------
def _stem(s: str) -> str:
    base = os.path.basename(s)
    return os.path.splitext(base)[0]

def _display_name_for_model_id(model_id: str, db: Dict) -> str:
    entry = db.get("models", {}).get(model_id, {})
    ck = entry.get("checkpoint_name", "")
    if ck:
        return _stem(ck)
    # Sinon essaie de rendre lisible
    up = model_id.upper()
    if up.startswith("MODEL_") or up.startswith("UNKNOWN_"):
        if "SDXL" in up:   return "SDXL_Model"
        if "FLUX" in up:   return "FLUX_Model"
        if "PONY" in up:   return "PONY_Model"
        if "ILLUST" in up: return "ILLUST_Model"
        if "SD" in up:     return "SD_Model"
        return f"Model_{model_id[-8:]}"
    return model_id

def _list_preset_choices() -> list:
    # Retourne ["none", "Display/PresetID", ...]
    choices = ["none"]
    models_root = os.path.join(PRESET_DIR, "models")
    if not os.path.isdir(models_root):
        return choices
    db = _load_model_database()
    for model_id in os.listdir(models_root):
        mdir = os.path.join(models_root, model_id)
        if not os.path.isdir(mdir):
            continue
        display = _display_name_for_model_id(model_id, db)
        for preset_id in os.listdir(mdir):
            pdir = os.path.join(mdir, preset_id)
            if not os.path.isdir(pdir):
                continue
            if os.path.exists(os.path.join(pdir, "preset.json")):
                choices.append(f"{display}/{preset_id}")
    return choices if len(choices) > 1 else ["none"]

# ------------------------------
# Le nœud : preview interne uniquement
# ------------------------------
class ModelPresetLoaderInternal:
    """
    Navigateur de presets avec **preview interne** (aucune sortie IMAGE).
    - Dropdown "preset_name" => "DisplayName/PresetID"
    - Dessin dans canvas via JS (contain fit, DPI-aware)
    - GET /model_preset_pilot/preview?preset_name=... pour servir l'image
    """

    aux_id = "NewLouwa/ComfyUI-Model_preset_Pilot/Internal"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force un refresh régulier pour mettre à jour la liste si des presets apparaissent
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset_name": (_list_preset_choices(),),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    # AUCUNE SORTIE : preview interne
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "run"
    CATEGORY = "🤖 Model Preset Pilot"

    # ---------- UI / JS ----------
    @classmethod
    def get_js(cls):
        # Canvas + redimensionnement dynamique (contain fit, devicePixelRatio),
        # redraw sur changement de preset et sur resize/zoom.
        js = r"""
        function setupMPPInternal(node) {
            // container de la carte
            const root = node.widgets?.[0]?.options?.el?.parentElement?.parentElement || node;
            root.style.position = "relative";

            // Canvas
            const canvas = document.createElement("canvas");
            canvas.style.position = "absolute";
            canvas.style.left = "8px";
            canvas.style.right = "8px";
            canvas.style.bottom = "8px";
            canvas.style.top = "64px";
            canvas.style.border = "1px solid #333";
            canvas.style.borderRadius = "8px";
            canvas.style.background = "#171717";
            root.appendChild(canvas);

            // Stocker la ref dans le node
            node.__mpp_canvas = canvas;

            // Texte d'info
            const info = document.createElement("div");
            info.style.position = "absolute";
            info.style.left = "8px";
            info.style.right = "8px";
            info.style.top = "8px";
            info.style.height = "44px";
            info.style.display = "flex";
            info.style.alignItems = "center";
            info.style.justifyContent = "center";
            info.style.fontFamily = "monospace";
            info.style.fontSize = "11px";
            info.style.color = "#c8c8c8";
            info.style.background = "rgba(0,0,0,0.35)";
            info.style.border = "1px solid #333";
            info.style.borderRadius = "6px";
            info.textContent = "Model Preset Pilot — Internal Preview";
            root.appendChild(info);

            // Helpers draw
            function drawPlaceholder(ctx, w, h) {
                ctx.clearRect(0,0,w,h);
                const grad = ctx.createLinearGradient(0,0,w,h);
                grad.addColorStop(0,"#1e1e1e");
                grad.addColorStop(1,"#121212");
                ctx.fillStyle = grad; ctx.fillRect(0,0,w,h);
                ctx.strokeStyle = "#333"; ctx.lineWidth = 1;
                ctx.strokeRect(0.5,0.5,w-1,h-1);
                ctx.fillStyle = "#777"; ctx.font = "12px sans-serif";
                ctx.textAlign = "center";
                ctx.fillText("No preview", w/2, h/2);
            }

            function resizeCanvas() {
                // Fit au style CSS + HiDPI
                const dpr = window.devicePixelRatio || 1;
                const rect = canvas.getBoundingClientRect();
                const w = Math.max(32, Math.floor(rect.width));
                const h = Math.max(32, Math.floor(rect.height));
                canvas.width  = Math.floor(w * dpr);
                canvas.height = Math.floor(h * dpr);
                const ctx = canvas.getContext("2d");
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                drawPlaceholder(ctx, w, h);
                return {ctx, w, h};
            }

            function updateFromPreset() {
                const presetWidget = node.widgets?.find(w => w.name === "preset_name");
                if (!presetWidget) return;
                const choice = presetWidget.value; // "Display/PresetID"
                const url = `/model_preset_pilot/preview?preset_name=${encodeURIComponent(choice)}&t=${Date.now()}`;

                const {ctx, w, h} = resizeCanvas();
                const img = new Image();
                img.onload = () => {
                    // Contain fit + letterbox
                    const r = Math.min(w / img.width, h / img.height);
                    const dw = Math.floor(img.width  * r);
                    const dh = Math.floor(img.height * r);
                    const x = Math.floor((w - dw) / 2);
                    const y = Math.floor((h - dh) / 2);
                    ctx.clearRect(0,0,w,h);
                    // fond
                    const grad = ctx.createLinearGradient(0,0,w,h);
                    grad.addColorStop(0,"#1e1e1e");
                    grad.addColorStop(1,"#101010");
                    ctx.fillStyle = grad; ctx.fillRect(0,0,w,h);
                    // cadre
                    ctx.strokeStyle = "#333"; ctx.lineWidth = 1; ctx.strokeRect(0.5,0.5,w-1,h-1);
                    // image
                    ctx.imageSmoothingEnabled = true;
                    ctx.imageSmoothingQuality = "high";
                    ctx.drawImage(img, x, y, dw, dh);
                };
                img.onerror = () => {
                    drawPlaceholder(ctx, w, h);
                };
                img.src = url;
            }

            // Brancher les callbacks de preset_name
            const presetWidget = node.widgets?.find(w => w.name === "preset_name");
            if (presetWidget) {
                const orig = presetWidget.callback;
                presetWidget.callback = function() {
                    if (orig) orig.apply(this, arguments);
                    updateFromPreset();
                };
            }

            // Resize/zoom handlers
            const ro = new ResizeObserver(() => updateFromPreset());
            ro.observe(canvas);
            window.addEventListener("resize", updateFromPreset);

            // First paint
            setTimeout(updateFromPreset, 200);
        }

        app.registerExtension({
            name: "ModelPresetPilot.InternalPreview",
            async nodeCreated(node) {
                if (node.comfyClass === "ModelPresetLoaderInternal") {
                    setTimeout(() => setupMPPInternal(node), 100);
                }
            }
        });
        """
        return js

    # ---------- Execution ----------
    def run(self, preset_name="none", unique_id=None, extra_pnginfo=None):
        # Aucune sortie, aucun side-effect global. Le canvas JS se charge via HTTP GET.
        # On peut néanmoins logger pour debug :
        print(f"[ModelPresetPilot] Internal preview request for preset: {preset_name}")
        return ()

# ------------------------------
# Node registration
# ------------------------------
NODE_CLASS_MAPPINGS = {
    "ModelPresetLoaderInternal": ModelPresetLoaderInternal
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelPresetLoaderInternal": "🧭 Model Preset Loader (Internal Preview)"
}
