"""SigLIP as a video frame backbone: the vision tower behind DeepTune's usual 2D backbone API, so frame pooling, single
frame, late fusion, CNN + RNN and the temporal transformer can use it. ``--model_version siglip`` loads the weights the
images modality uses (src/vision/siglip.py); ``siglip_tiny`` is untrained and tiny, for tests and CPU smoke runs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGLIP_INPUT_SIZE = {'siglip': 384, 'siglip_tiny': 32}


def _load_vision_tower(version: str) -> nn.Module:
    if version == 'siglip':
        from src.vision.siglip import load_siglip_model_offline
        return load_siglip_model_offline().vision_model
    if version == 'siglip_tiny':
        from transformers import SiglipVisionConfig, SiglipVisionModel
        config = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                                    image_size=SIGLIP_INPUT_SIZE[version], patch_size=8)
        return SiglipVisionModel(config)
    raise ValueError(f"Invalid SigLIP version {version!r}. Choose 'siglip' or 'siglip_tiny'.")


class SiglipClipTransform:
    """SigLIP's preprocessing for a uint8 clip ``[T, C, H, W]``: bicubic resize, then mean and std 0.5."""

    def __init__(self, version: str):
        if version not in SIGLIP_INPUT_SIZE:
            raise ValueError(f"Invalid SigLIP version {version!r}. Choose 'siglip' or 'siglip_tiny'.")
        self.size = SIGLIP_INPUT_SIZE[version]

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        frames = F.interpolate(clip.float() / 255.0, size=(self.size, self.size), mode='bicubic', align_corners=False)
        return (frames.clamp(0.0, 1.0) - 0.5) / 0.5


class adjustedSiglip(nn.Module):
    """Same signature, attributes and ``forward(x, extract_embed=False)`` as the adjusted*Net classes in src/vision."""

    def __init__(self, num_classes, siglip_version, added_layers=2, embedding_layer_size=1000, freeze_backbone=False,
                 task_type='cls', output_dim=1):
        super().__init__()
        assert task_type in ['cls', 'reg'], "task_type must be 'cls' or 'reg'"
        self.added_layers = added_layers
        self.model = _load_vision_tower(siglip_version)
        self.fc1_input = self.model.config.hidden_size
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        out = num_classes if task_type == 'cls' else output_dim
        if added_layers == 2:
            self.fc1 = nn.Linear(self.fc1_input, embedding_layer_size)
            self.fc2 = nn.Linear(embedding_layer_size, out)
        elif added_layers == 1:
            self.fc1 = nn.Linear(self.fc1_input, out)
        else:
            self.fc1 = None

    def _features(self, x):
        return self.model(pixel_values=x).pooler_output

    def forward(self, x, extract_embed=False):
        x = self._features(x)
        if extract_embed:
            return self.fc1(x) if self.added_layers == 2 else x
        if self.added_layers == 2:
            return self.fc2(F.relu(self.fc1(x)))
        return self.fc1(x) if self.added_layers == 1 else x


class adjustedPeftSiglip(adjustedSiglip):
    """LoRA on the attention's query and value projections (the targets of src/vision/siglip.py). As in the other PEFT
    classes, `embedding_layer_size` is the LoRA rank."""

    def __init__(self, num_classes, siglip_version, added_layers=2, embedding_layer_size=1000, freeze_backbone=False,
                 task_type='cls', output_dim=1):
        from peft import LoraConfig, get_peft_model
        super().__init__(num_classes, siglip_version, added_layers, embedding_layer_size, freeze_backbone, task_type,
                         output_dim)
        config = LoraConfig(r=embedding_layer_size, lora_alpha=16, lora_dropout=0.1, bias='none',
                            target_modules=['q_proj', 'v_proj'])
        self.peftmodel = get_peft_model(self.model, config)

    def _features(self, x):
        return self.peftmodel(pixel_values=x).pooler_output
