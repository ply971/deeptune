import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from src.video.video3d import _build_backbone


def lora_target_modules(model):
    """Names of the modules that get a LoRA adapter.

    Conv3d covers the 3D-CNN backbones (r3d_18, mc3_18, r2plus1d_18) and Linear
    the video-transformer backbones (mvit_v2_s, swin3d_*). The check is
    isinstance, because torchvision's 3D ResNets build their convs from
    subclasses (Conv3DSimple, Conv3DNoTemporal); an exact type check reached
    only the stem and downsample convs, 4 of r3d_18's 20 (V-007). Grouped convs
    are skipped, as is any conv subclass with a forward of its own, which
    plain_conv_subclasses cannot convert. So are the layers whose weights their
    parent reads directly, which an adapter could never affect: the Linear
    subclass inside nn.MultiheadAttention and Swin's attention qkv and proj.
    """
    direct = directly_read_swin_layers(model)
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            if module.groups != 1 or _plain_conv_base(module) is None:
                continue
        elif not isinstance(module, nn.Linear) or isinstance(module, NonDynamicallyQuantizableLinear):
            continue
        if name not in direct:
            targets.append(name)
    return targets


def directly_read_swin_layers(model):
    """Names of Swin's attention qkv and proj. torchvision's ShiftedWindowAttention
    (2D, 2D V2 and 3D) passes self.qkv.weight and self.proj.weight to F.linear and
    never calls the layers, so a LoRA adapter on them gets no gradient."""
    names = set()
    for name, module in model.named_modules():
        if type(module).__name__.startswith('ShiftedWindowAttention'):
            names.update(f'{name}.{child}' if name else child for child in ('qkv', 'proj'))
    return names


def _plain_conv_base(module):
    """nn.Conv2d or nn.Conv3d if ``module`` computes exactly what that class does, else None."""
    base = nn.Conv3d if isinstance(module, nn.Conv3d) else nn.Conv2d
    same = type(module).forward is base.forward and type(module)._conv_forward is base._conv_forward
    return base if same else None


def plain_conv_subclasses(model):
    """Turn conv subclasses into their plain base class, in place.

    PEFT builds each adapter with ``type(base_layer)(in, r, kernel, stride,
    padding, bias=False)``, which Conv3DSimple's constructor (in_planes,
    out_planes, midplanes, stride, padding) rejects. These subclasses only
    differ in that constructor, so the class change alters neither the weights
    nor the forward pass, and the module names stay the same.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)) and type(module) not in (nn.Conv2d, nn.Conv3d):
            base = _plain_conv_base(module)
            if base is not None:
                module.__class__ = base


def upgrade_legacy_lora_state(model, state_dict):
    """Convert weights saved with older LoRA targets to this model's layout.

    Two changes are handled, and neither alters the old model's predictions:

    * Layers that now have an adapter (V-007: those runs had LoRA on the stem and
      downsample convs only). The state dict holds plain ``X.weight`` where this
      model has ``X.base_layer.weight`` and ``X.lora_A/B``. The plain weights
      move to ``base_layer`` and the new adapters are stored with lora_B = 0, so
      the delta is exactly zero.
    * Layers that no longer have one (Swin's attention qkv and proj, V-040). The
      state dict holds ``X.base_layer.weight`` and ``X.lora_A/B`` where this
      model has plain ``X.weight``. Those adapters never ran, so they are
      dropped and the weights move back.

    Returns (state dict, number of upgraded modules); a state dict with the
    current layout comes back as it is.
    """
    current = model.state_dict()
    upgraded = dict(state_dict)
    added, removed = set(), set()
    for key in current:
        if '.base_layer.' not in key or key in state_dict:
            continue
        legacy_key = key.replace('.base_layer.', '.')
        if legacy_key in upgraded:
            upgraded[key] = upgraded.pop(legacy_key)
            added.add(key.split('.base_layer.')[0])
    for key in state_dict:
        if '.base_layer.' not in key or key in current:
            continue
        plain_key = key.replace('.base_layer.', '.')
        if plain_key in current:
            upgraded[plain_key] = upgraded.pop(key)
            removed.add(key.split('.base_layer.')[0])
    if not added and not removed:
        return state_dict, 0
    for key, value in current.items():
        if '.lora_' in key and key.split('.lora_')[0] in added and key not in upgraded:
            upgraded[key] = torch.zeros_like(value) if '.lora_B.' in key else value
    for key in state_dict:
        if '.lora_' in key and key.split('.lora_')[0] in removed:
            del upgraded[key]
    # The state dict holds each layer twice, under model. and under peftmodel.
    return upgraded, len({name.replace('peftmodel.base_model.model.', 'model.') for name in added | removed})


class adjustedPeftVideo3D(nn.Module):

    def __init__(self, num_classes, video_version, added_layers, lora_attention_dimension, freeze_backbone=False, task_type="cls", output_dim=1, lora_rank=None):
        """

        Customised native video classifier applying Parameter Efficient Fine
        Tuning with LoRA as part of DeepTune's proposed adjustments, mirroring
        adjustedPeftResNet but backed by torchvision's spatio-temporal video
        architectures (see src/video/video3d.py for the non-PEFT variant and
        VIDEO3D_ARCHITECTURES for the supported video_version values).

        Args:
            num_classes (int): Number of classes in your dataset.
            video_version (str): One of VIDEO3D_ARCHITECTURES, e.g. "r3d_18".
            added_layers (int): Number of additional layers you want to add while finetuning your model.
            lora_attention_dimension (int): If you chose added_layers to be 2, this specifies the size of the intermediate layer in between (--embed_size).
            freeze_backbone (bool): Determine whether you want to apply transfer learning on the backbone weights or the whole model.
            task_type (str): Determine whether you want classification or regression.
            output_dim (int): The dimension of the output of the regression model, default = 1.
            lora_rank (int): The LoRA rank (--lora_rank). None means lora_attention_dimension, which
                is what runs recorded before --lora_rank existed were built with.

        """

        super(adjustedPeftVideo3D, self).__init__()

        assert task_type in ["cls", "reg"], "task_type must be 'cls' or 'reg'"

        self.num_classes = num_classes
        self.added_layers = added_layers
        self.lora_attention_dimension = lora_attention_dimension
        self.lora_rank = lora_attention_dimension if lora_rank is None else lora_rank
        self.freeze_backbone = freeze_backbone
        self.video_version = video_version
        self.task_type = task_type
        self.output_dim = output_dim

        self.model, self.fc_in = _build_backbone(video_version)
        self.flatten = nn.Flatten()

        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Backbone Parameters are frozen!')

        if self.added_layers == 2:
            self.fc1 = nn.Linear(self.fc_in, self.lora_attention_dimension)

            if self.task_type == "cls":
                self.fc2 = nn.Linear(self.lora_attention_dimension, self.num_classes)
            else:
                self.fc2 = nn.Linear(self.lora_attention_dimension, self.output_dim)
        elif self.added_layers == 1:

            if self.task_type == "cls":
                self.fc1 = nn.Linear(self.fc_in, self.num_classes)
            else:
                self.fc1 = nn.Linear(self.fc_in, self.output_dim)
        else:
            self.fc1 = None

        # Apply PEFT

        self.peftmodel = self.applyPEFT(self.model)

    def applyPEFT(self, model):

        """
        Apply PEFT with LoRA on the customised model.

        Arguments:
            model (torchvision.models.video): The adjusted video backbone before applying PEFT on.

        Returns:

            peft_model: PEFTed adjusted video backbone.
        """

        target_modules = lora_target_modules(model)
        plain_conv_subclasses(model)
        print('Target Modules', target_modules)
        not_adapted = directly_read_swin_layers(model)
        if not_adapted:
            print(f"LoRA does not adapt Swin's attention: torchvision reads the weights of its {len(not_adapted)} qkv "
                  'and proj layers directly, so an adapter there would do nothing.')

        self.lora_config = LoraConfig(r=self.lora_rank, lora_alpha=16, lora_dropout=0.1, bias="none", target_modules=target_modules)
        print(self.lora_config)

        peft_model = get_peft_model(model, self.lora_config)
        return peft_model

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Also loads weights saved with older LoRA targets (see
        upgrade_legacy_lora_state); a checkpoint that fits neither layout gets
        an error naming the likely reasons."""
        state_dict, upgraded = upgrade_legacy_lora_state(self, state_dict)
        if upgraded:
            print(f'Checkpoint from before the LoRA coverage fix: {upgraded} layers changed. New adapters are '
                  'initialised to no effect and adapters that never ran are dropped, so predictions match the old model.')
        try:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        except RuntimeError as exc:
            raise RuntimeError(
                f'These weights do not fit the {self.video_version} PEFT model. Check --model_version, --added_layers, '
                f'--embed_size, --lora_rank (runs recorded without it used --embed_size as the rank) and --num_classes against the run that saved them.\n{exc}') from exc

    def forward(self, x, extract_embed=False):
        """
        After applying PEFT, and according to the number of added_layers, we apply the forward pass.

        Note: Unlike the non-PEFT version, if added_layers = 1 this wouldn't return the same
        embeddings as the pretrained backbone because applying PEFT with LoRA has altered the
        network's weights.
        """
        # x: [B, T, C, H, W] -> torchvision's video models expect [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        x = self.peftmodel(x)
        x = self.flatten(x)

        if self.added_layers == 1 and extract_embed:
            return x

        elif self.added_layers == 2 and extract_embed:
            x = self.fc1(x)
            return x

        if self.added_layers == 2:
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)

        elif self.added_layers == 1:
            x = self.fc1(x)

        return x
