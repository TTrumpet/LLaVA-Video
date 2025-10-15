import torch
import torch.nn as nn
from transformers import CLIPImageProcessor
from vit_rope2d_hf import MLCDVisionModel

# NOTE: Follow the instructions from the original repo to download unicom repository into the main directory

class MvtVisionTower(nn.Module):
    def __init__(self, vision_tower_path, **kwargs):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_path = vision_tower_path

        # Get hidden size from a temporary config load
        # This prevents loading the whole model just to get this value
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(vision_tower_path)
        self.hidden_size = config.hidden_size

        # Placeholder for the model and processor until load_model() is called
        self.vision_tower = None
        self.image_processor = None

    def load_model(self, device_map=None):
        """Load the model and processor."""
        if self.is_loaded:
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_path)
        self.vision_tower = MLCDVisionModel.from_pretrained(self.vision_tower_path, device_map=device_map)
        self.vision_tower.eval() # Set to evaluation mode

        self.is_loaded = True

    def forward(self, images):
        # This is the same forward pass logic as before
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        # Select the feature layer LLaVA expects
        image_features = image_forward_outs.hidden_states[-2] # e.g., second to last layer
        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device
