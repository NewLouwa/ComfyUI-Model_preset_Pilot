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
        try:
            # 1️⃣ Clamp values to avoid out-of-range values
            images = images.clamp(0, 1)

            # 2️⃣ Convert to uint8 for display
            np_images = (images.cpu().numpy() * 255).astype(np.uint8)

            # 3️⃣ Send to web client for display
            comfy.utils.save_images(np_images)

        except AttributeError:
            # Fallback if comfy.utils.save_images doesn't exist
            print(f"Preview: Displaying {images.shape} image tensor")
        except Exception as e:
            print(f"Warning: Could not preview image: {e}")

        # This node returns nothing, it acts as a "visual terminal"
        return ()


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "PreviewImage": PreviewImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewImage": "🖼️ Preview Image",
}
