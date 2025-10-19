# Preview Image Node
# Custom ComfyUI node for internal image preview
# Based on ComfyUI's official PreviewImage node

import torch
import numpy as np
import comfy.utils
from PIL import Image


class PreviewImage:
    """
    Internal Preview Image - displays images in ComfyUI interface
    Based on ComfyUI's official PreviewImage node from nodes/image_nodes.py
    """
    
    aux_id = "NewLouwa/ComfyUI-Model_preset_Pilot"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "preview"
    CATEGORY = "🤖 Model Preset Pilot"

    def preview(self, images):
        """
        Preview images in ComfyUI interface
        Args:
            images: torch.Tensor [B,H,W,C] float32 in range [0,1]
        """
        # 1️⃣ Clamp values to avoid out-of-range values
        images = images.clamp(0, 1)

        # 2️⃣ Convert to uint8 for display
        np_images = (images.cpu().numpy() * 255).astype(np.uint8)

        # 3️⃣ Send to web client for display
        comfy.utils.save_images(np_images)

        # This node returns nothing, it acts as a "visual terminal"
        return ()


class LoadLocalImage:
    """
    Load Local Image - loads images from custom directories
    Based on ComfyUI's official LoadImage node
    """
    
    aux_id = "NewLouwa/ComfyUI-Model_preset_Pilot"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "data/defaults/NothingHere_Robot.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "🤖 Model Preset Pilot"

    def load_image(self, image_path):
        """
        Load image from local path
        Args:
            image_path: str - path to image file
        Returns:
            torch.Tensor [1,H,W,C] - image tensor
        """
        try:
            # 1️⃣ Open and convert to RGB
            pil_image = Image.open(image_path).convert("RGB")
            
            # 2️⃣ Convert to numpy then tensor [1,H,W,C]
            arr = np.array(pil_image)
            t = torch.from_numpy(arr).float() / 255.0
            return (t.unsqueeze(0),)
            
        except Exception as e:
            print(f"Warning: Could not load image from {image_path}: {e}")
            # Return a default black image
            default_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (default_tensor,)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PreviewImage": PreviewImage,
    "LoadLocalImage": LoadLocalImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewImage": "🖼️ Preview Image",
    "LoadLocalImage": "📂 Load Local Image",
}
