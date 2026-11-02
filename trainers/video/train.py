from pathlib import Path
from torch.utils.data import DataLoader

from helpers import transformations
from trainers.vision.trainer import Trainer
from options import UNIQUE_ID, DEVICE
from datasets.video_datasets import ParquetVideoDataset, native_video_transform, video_loader_kwargs, video_options_from_args
from cli import DeepTuneVisionOptions
from utils import RunType, set_seed, build_video_model, is_native_video_architecture, resolve_lora_rank
from trainers.param_budget import build_with_budget
from src.video.multi_clip import wrap_for_test_clips
from handlers.training_args import TrainingOptions
from trainers.training_basics import data_fingerprint, is_pretrained_version, seed_everything


def main():
    args = DeepTuneVisionOptions(RunType.TRAIN)
    TRAIN_PATH: Path = args.train_df
    VAL_PATH: Path = args.val_df
    MODE = args.mode
    NUM_CLASSES = args.num_classes
    OUT = args.out

    MODEL_VERSION = args.model_version
    MODEL_STR = args.model

    ADDED_LAYERS = args.added_layers
    EMBED_SIZE = args.embed_size
    FREEZE_BACKBONE = args.freeze_backbone
    USE_PEFT = args.use_peft

    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    FIXED_SEED = args.fixed_seed
    NUM_FRAMES = args.num_frames
    POOLING = args.pooling

    train(
        train_df=TRAIN_PATH,
        val_df=VAL_PATH,
        mode=MODE,
        num_classes=NUM_CLASSES,
        out=OUT,
        model_version=MODEL_VERSION,
        model_str=MODEL_STR,
        added_layers=ADDED_LAYERS,
        embed_size=EMBED_SIZE,
        freeze_backbone=FREEZE_BACKBONE,
        use_peft=USE_PEFT,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fixed_seed=FIXED_SEED,
        num_frames=NUM_FRAMES,
        pooling=POOLING,
        args=args,
        trainable_params=args.trainable_params,
    )


def train(
        train_df: Path,
        val_df: Path,
        out: Path,
        freeze_backbone: bool,
        use_peft: bool,
        fixed_seed: int,
        mode: str,
        model_version: str,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        added_layers: int,
        num_classes: int,
        embed_size: int,
        model_str: str,
        args: DeepTuneVisionOptions,
        num_frames: int = 8,
        pooling: str = "mean",
        trainable_params: int | None = None,
):
    """
    Trains a DeepTune video classifier - either one of DeepTune's native
    spatio-temporal architectures (e.g. "r3d_18") or a frame-sampling model
    built on one of the existing 2D vision backbones (e.g. "resnet50"),
    depending on `model_version`. See utils.build_video_model for how that
    choice is resolved.

    This mirrors trainers/vision/train.py's `train`, reusing the exact same
    Trainer class - both model families expose forward(x) -> [B, num_classes],
    so no video-specific training loop is needed.

    `trainable_params` (optional) caps how many parameters are trained - see
    trainers/param_budget.py. With PEFT it picks the LoRA rank, which is then
    written back to `args` before the arguments are saved: as `lora_rank` on
    native models, as `embed_size` on the others, whose head width is the rank.
    """

    if fixed_seed:
        set_seed(fixed_seed)
    # --- training basics (T1) ---
    training_options = TrainingOptions.from_args(args).validate(mode)
    if training_options.seed is not None:
        seed_everything(training_options.seed)
    # --- end training basics ---

    if added_layers == 0:
        raise ValueError('As you apply one of transfer learning or PEFT, please choose 1 or 2 as your preferred number of added_layers.')

    TRAINVAL_OUTPUT_DIR = (out / f"trainval_output_{model_str}_{UNIQUE_ID}")

    # Either storage format (file paths or clip bytes) is detected by from_parquet.
    # Training samples randomly (--temporal_sampling, --video_augment); validation
    # uses the deterministic evaluation sampling with one clip per video.
    video_options = video_options_from_args(args)
    clip_transform = native_video_transform(model_version)
    train_dataset = ParquetVideoDataset.from_parquet(parquet_file=train_df, transform=transformations, num_frames=num_frames,
                                                     clip_transform=clip_transform, train=True, options=video_options)
    val_dataset = ParquetVideoDataset.from_parquet(parquet_file=val_df, transform=transformations, num_frames=num_frames,
                                                   clip_transform=clip_transform, train=False, options=video_options)

    train_kwargs = video_loader_kwargs(train_dataset, batch_size, True, video_options.num_workers)
    val_kwargs = video_loader_kwargs(val_dataset, batch_size, False, video_options.num_workers, role='val')
    if training_options.checkpoints_or_resumes:
        # An exact resume needs fresh workers each epoch: a persistent worker's random
        # state cannot be saved, and it draws no new base seed from the main generator.
        train_kwargs['persistent_workers'] = val_kwargs['persistent_workers'] = False
    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)

    lora_rank = resolve_lora_rank(args) if use_peft and is_native_video_architecture(model_version) else None
    if trainable_params is None:
        model = build_video_model(
            model_version=model_version,
            num_classes=num_classes,
            added_layers=added_layers,
            embed_size=embed_size,
            freeze_backbone=freeze_backbone,
            mode=mode,
            use_peft=use_peft,
            pooling=pooling,
            **({} if lora_rank is None else {'lora_rank': lora_rank}),
        )
    elif use_peft and is_native_video_architecture(model_version):
        def build_at_rank(rank):
            return build_video_model(
                model_version=model_version, num_classes=num_classes, added_layers=added_layers,
                embed_size=embed_size, freeze_backbone=freeze_backbone, mode=mode, use_peft=True, pooling=pooling,
                lora_rank=rank,
            )
        model, args.lora_rank = build_with_budget(build_at_rank, trainable_params, True, embed_size, freeze_backbone,
                                                  rank_is_head_width=False)
    else:
        def build(rank):
            return build_video_model(
                model_version=model_version, num_classes=num_classes, added_layers=added_layers,
                embed_size=rank, freeze_backbone=freeze_backbone, mode=mode, use_peft=use_peft, pooling=pooling,
            )
        model, embed_size = build_with_budget(build, trainable_params, use_peft, embed_size, freeze_backbone)
        args.embed_size = embed_size

    TRAINVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, train_loader=train_loader, val_loader=val_loader, learning_rate=learning_rate, mode=mode, num_epochs=num_epochs, output_dir=TRAINVAL_OUTPUT_DIR, training_options=training_options,
                      backbone_pretrained=is_pretrained_version(model_version))
    trainer.fit_error_model = not getattr(args, 'no_error_model', False)
    trainer.data_identity.update(train_file=data_fingerprint(train_df), val_file=data_fingerprint(val_df))
    # The error model must see what evaluation sees: with --test_clips N it is
    # fitted on N-clip averaged validation probabilities (A2 V-016), and with
    # --test_crops K on the K-crop average inside each clip.
    trainer.error_fit_info = {'test_clips': video_options.test_clips, 'test_crops': video_options.test_crops}
    if video_options.test_crops > 1 and video_options.test_clips == 1 and trainer.fit_error_model:
        trainer.error_fit_model = wrap_for_test_clips(model, 1, mode, test_crops=video_options.test_crops)
    if video_options.test_clips > 1 and trainer.fit_error_model:
        fit_dataset = ParquetVideoDataset.from_parquet(parquet_file=val_df, transform=transformations, num_frames=num_frames,
                                                       clip_transform=clip_transform, train=False, options=video_options,
                                                       num_clips=video_options.test_clips)
        trainer.error_fit_loader = DataLoader(fit_dataset, **video_loader_kwargs(fit_dataset, batch_size, False,
                                                                                 video_options.num_workers))
        trainer.error_fit_model = wrap_for_test_clips(model, video_options.test_clips, mode, test_crops=video_options.test_crops)

    print('The Trainer class is loaded successfully.')

    trainer.train()
    trainer.validate()

    print('Saving the model and arguments is under way!')

    output_dir = f'{TRAINVAL_OUTPUT_DIR}/model_weights.pth'

    trainer.saveModel(path=output_dir)
    from src.video.feature_backbones import record_weights
    record_weights(args, model_version)
    args.save_args(TRAINVAL_OUTPUT_DIR)

    return output_dir


if __name__ == "__main__":
    main()
