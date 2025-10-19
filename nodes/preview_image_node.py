# Preview Image Node
# Simple internal preview node for displaying images in ComfyUI interface

import torch
import comfy.utils
import numpy as np


class PreviewImage:
    """
    Internal Preview Image Node
    Displays any IMAGE tensor directly in the ComfyUI interface
    """
    
    aux_id = "NewLouwa/ComfyUI-Model_preset_Pilot"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "preview"
    CATEGORY = "🤖 Model Preset Pilot"

    def preview(self, image: torch.Tensor):
        """
        Display image tensor in ComfyUI interface
        
        Args:
            image: Tensor [B,H,W,C] with values in [0,1] range
        """
        # 1️⃣ Clamp values to ensure they're in [0,1] range
        image = image.clamp(0, 1)
        
        # 2️⃣ Convert to uint8 for display
        np_images = (image.cpu().numpy() * 255).astype(np.uint8)
        
        # 3️⃣ Send to ComfyUI frontend for display
        comfy.utils.save_images(np_images)
        
        # This node doesn't return anything - it's a visual terminal
        return ()


# Register the node
NODE_CLASS_MAPPINGS = {
    "PreviewImage": PreviewImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewImage": "🖼️ Preview Image"
}
