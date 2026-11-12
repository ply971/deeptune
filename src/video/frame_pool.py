import torch
import torch.nn as nn
import torch.nn.functional as F

# Value of the `pool_format` buffer: pooled embeddings. Old checkpoints have no such key.
EMBEDDING_POOL_FORMAT = 2


class VideoFramePoolClassifier(nn.Module):
    """
    Wraps one of DeepTune's existing 2D vision backbones (ResNet, EfficientNet,
    DenseNet, Swin, ViT, ConvNeXt, VGG - PEFT or not) to classify video clips.

    Each sampled frame of the clip is passed through the *same* backbone
    (weights are shared across frames - it is applied frame-by-frame, not one
    backbone per frame), producing one embedding per frame. Those are then
    pooled over time - either by a simple average, or by a small learned
    attention layer that lets the model weigh more informative frames more
    heavily - and the backbone's head classifies the pooled clip embedding.

    Checkpoints from before embeddings were pooled (no `pool_format` key) pooled
    the per-frame *logits*. `load_state_dict` detects them and keeps that
    behaviour for that model, so they still load and predict exactly as before.

    This keeps video classification fully compatible with DeepTune's existing
    Trainer/TestTrainer pipeline (forward(x) -> [B, num_classes]) and PEFT/LoRA
    infrastructure, without introducing a whole separate spatio-temporal model
    zoo. See src/video/video3d.py for DeepTune's native 3D-CNN/video-transformer
    alternative.

    Args:
        frame_model_cls (Type[nn.Module]): One of DeepTune's adjusted*Net classes
            (e.g. adjustedResNet or adjustedPeftResNet) used to build the shared
            per-frame backbone. Every one of these classes accepts the same
            positional constructor signature: (num_classes, model_version,
            added_layers, embedding_layer_size, freeze_backbone, task_type=...).
        frame_model_version (str): Version string of the underlying 2D architecture
            (e.g. "resnet50").
        num_classes (int): Number of classes in your dataset.
        added_layers (int): Number of additional layers on the per-frame backbone.
        embedding_layer_size (int): Size of the intermediate embedding layer if
            added_layers=2 (or the LoRA attention dimension, for PEFT backbones).
        freeze_backbone (bool): Whether to freeze the per-frame backbone's weights.
        task_type (str): "cls" or "reg".
        pooling (str): "mean" or "attention" - how per-frame embeddings are combined
            over time into a single clip embedding.
    """

    def __init__(
        self,
        frame_model_cls,
        frame_model_version: str,
        num_classes: int,
        added_layers: int = 2,
        embedding_layer_size: int = 1000,
        freeze_backbone: bool = False,
        task_type: str = "cls",
        pooling: str = "mean",
    ):
        super().__init__()

        assert pooling in ("mean", "attention"), "pooling must be 'mean' or 'attention'"
        assert task_type in ("cls", "reg"), "task_type must be 'cls' or 'reg'"

        self.pooling = pooling
        self.task_type = task_type
        self.added_layers = added_layers
        self.num_classes = num_classes

        # Every adjusted*Net class (PEFT or not) shares this exact positional
        # signature - it's how DeepTune's own vision trainer instantiates them
        # generically regardless of architecture.
        self.backbone = frame_model_cls(
            num_classes, frame_model_version, added_layers, embedding_layer_size, freeze_backbone, task_type=task_type,
        )

        self.logit_pooling = False
        self.register_buffer("pool_format", torch.tensor(EMBEDDING_POOL_FORMAT))

        if pooling == "attention":
            self.attention = nn.Linear(self.embedding_dim, 1)

    @property
    def embedding_dim(self) -> int:
        """Size of what the backbone returns for extract_embed=True: fc1's output with two added layers, else fc1's input."""
        if self.added_layers == 2:
            return self.backbone.fc1.out_features
        return self.backbone.fc1.in_features

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + "pool_format" not in state_dict and not self.logit_pooling:
            self._use_logit_pooling()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _use_logit_pooling(self):
        """The old checkpoint format: no marker in the state dict (which keeps its fingerprint, so the run's
        error model still applies), and attention over per-frame outputs."""
        self.logit_pooling = True
        del self._buffers["pool_format"]
        if self.pooling == "attention":
            weight = self.attention.weight
            per_frame_out_dim = self.num_classes if self.task_type == "cls" else 1
            self.attention = nn.Linear(per_frame_out_dim, 1).to(weight.device, weight.dtype)

    def _classify(self, embedding):
        """The backbone's own head, as its forward() applies it after extract_embed's output."""
        if self.added_layers == 2:
            return self.backbone.fc2(F.relu(embedding))
        return self.backbone.fc1(embedding) if self.added_layers == 1 else embedding

    def _pool(self, per_frame):
        if self.pooling == "mean":
            return per_frame.mean(dim=1)
        weights = torch.softmax(self.attention(per_frame), dim=1)  # [B, T, 1]
        return (per_frame * weights).sum(dim=1)

    def forward(self, x, extract_embed=False):
        """
        Args:
            x (Tensor): Video clip batch of shape [B, T, C, H, W].
            extract_embed (bool): If True, returns the clip embedding the head
                sees (pooled with the model's own pooling) instead of a prediction.

        Returns:
            Tensor: [B, num_classes] (or [B, 1] for regression), or
                [B, embedding_dim] if extract_embed=True.
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        if self.logit_pooling:
            out = self.backbone(x, extract_embed=extract_embed).reshape(B, T, -1)
            return out.mean(dim=1) if extract_embed else self._pool(out)

        embedding = self._pool(self.backbone(x, extract_embed=True).reshape(B, T, -1))
        return embedding if extract_embed else self._classify(embedding)
