import torch.nn as nn


class VideoSingleFrameClassifier(nn.Module):
    """
    The "single frame" baseline of Karpathy et al. (CVPR 2014, "Large-scale
    Video Classification with Convolutional Neural Networks"): a 2D image
    backbone sees only the *middle* frame of the clip. Whatever it learns is
    appearance, never motion, so on a task where motion is the whole signal
    (the Phase 2 direction benchmark) it must stay at chance - that is the
    point of the baseline.

    Wraps any of DeepTune's adjusted*Net 2D backbones (PEFT or not) exactly
    like VideoFramePoolClassifier does, and keeps the video model contract:
    input clips [B, T, C, H, W], forward(x, extract_embed=False) returns
    [B, num_classes] (cls), [B, 1] (reg) or the middle frame's embedding.

    Args:
        frame_model_cls (Type[nn.Module]): adjusted*Net class used to build the
            2D backbone (positional signature (num_classes, model_version,
            added_layers, embedding_layer_size, freeze_backbone, task_type=...)).
        frame_model_version (str): e.g. "resnet18".
        num_classes (int): Number of classes in your dataset.
        added_layers (int): Number of additional layers on the backbone.
        embedding_layer_size (int): Size of the intermediate embedding layer
            (or the LoRA attention dimension, for PEFT backbones).
        freeze_backbone (bool): Whether to freeze the backbone's weights.
        task_type (str): "cls" or "reg".
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
    ):
        super().__init__()
        assert task_type in ("cls", "reg"), "task_type must be 'cls' or 'reg'"
        self.task_type = task_type
        self.added_layers = added_layers
        self.backbone = frame_model_cls(
            num_classes, frame_model_version, added_layers, embedding_layer_size, freeze_backbone, task_type=task_type,
        )

    @staticmethod
    def middle_index(num_frames: int) -> int:
        """Index of the frame the model looks at: T // 2 (the later of the two
        central frames when T is even)."""
        return num_frames // 2

    def forward(self, x, extract_embed=False):
        """
        Args:
            x (Tensor): Video clip batch of shape [B, T, C, H, W].
            extract_embed (bool): If True, returns the middle frame's embedding.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected a clip batch shaped [B, T, C, H, W], got {tuple(x.shape)}.")
        frame = x[:, self.middle_index(x.shape[1])]
        return self.backbone(frame, extract_embed=extract_embed)
