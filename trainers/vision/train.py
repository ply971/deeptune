from pathlib import Path
from torch.utils.data import DataLoader

from helpers import transformations
from trainers.vision.trainer import Trainer
from options import UNIQUE_ID, DEVICE, NUM_WORKERS, PERSIST_WORK, PIN_MEM
from datasets.image_datasets import ParquetImageDataset
from trainers.vision.custom_siglip_train import train_siglip
from cli import DeepTuneVisionOptions
from utils import get_model_cls, RunType,set_seed
from trainers.param_budget import build_with_budget
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
        mode:str,
        model_version:str,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        added_layers: int,
        num_classes: int,
        embed_size: int,
        model_str: str,
        args: DeepTuneVisionOptions,
        trainable_params: int | None = None,
):
    """Trains an image model. `trainable_params` (optional) caps how many
    parameters are trained - see trainers/param_budget.py; with PEFT the chosen
    LoRA rank is written back to `args.embed_size` before the arguments are saved."""

    MODEL_ARCHITECTURE = args.model_architecture
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

    if MODEL_ARCHITECTURE.lower() == "siglip" and model_version == "siglip":
        if trainable_params is not None:
            raise ValueError('--trainable_params is not supported for SigLIP; remove the budget or choose another model.')
        if training_options.non_default():
            raise ValueError(f"These flags are not supported for SigLIP, which has its own training loop: "
                             f"{', '.join('--' + name for name in training_options.non_default())}.")
        _ = train_siglip(
            num_classes=num_classes,
            added_layers=added_layers,
            embed_size=embed_size,
            train_dataset_path=train_df,
            val_dataset_path=val_df,
            outdir=TRAINVAL_OUTPUT_DIR,
            device=DEVICE,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            use_peft=use_peft,
            freeze_backbone=freeze_backbone,
        )
        args.save_args(TRAINVAL_OUTPUT_DIR)
        return TRAINVAL_OUTPUT_DIR

    train_dataset = ParquetImageDataset.from_parquet(parquet_file=train_df, transform=transformations)
    val_dataset = ParquetImageDataset.from_parquet(parquet_file=val_df, transform=transformations)

    # An exact resume needs fresh workers each epoch: a persistent worker's random
    # state cannot be saved, and it draws no new base seed from the main generator.
    persistent_workers = PERSIST_WORK and not training_options.checkpoints_or_resumes
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEM,
        persistent_workers=persistent_workers
    )

    chosen_model = get_model_cls(model_architecture=MODEL_ARCHITECTURE, use_peft=use_peft)
    if trainable_params is None:
        model = chosen_model(num_classes, model_version, added_layers, embed_size, freeze_backbone, task_type=mode)
    else:
        def build(rank):
            return chosen_model(num_classes, model_version, added_layers, rank, freeze_backbone, task_type=mode)
        model, embed_size = build_with_budget(build, trainable_params, use_peft, embed_size, freeze_backbone)
        args.embed_size = embed_size

    TRAINVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    trainer = Trainer(model, train_loader=train_loader, val_loader=val_loader, learning_rate=learning_rate, mode=mode, num_epochs=num_epochs, output_dir=TRAINVAL_OUTPUT_DIR, training_options=training_options,
                      backbone_pretrained=is_pretrained_version(model_version))
    trainer.fit_error_model = not getattr(args, 'no_error_model', False)
    trainer.data_identity.update(train_file=data_fingerprint(train_df), val_file=data_fingerprint(val_df))
    
    print('The Trainer class is loaded successfully.')
    
    trainer.train()
    trainer.validate()
    
    print('Saving the model and arguments is under way!')

    output_dir = f'{TRAINVAL_OUTPUT_DIR}/model_weights.pth'
    
    trainer.saveModel(path=output_dir)
    args.save_args(TRAINVAL_OUTPUT_DIR)

    return output_dir
    


if __name__ == "__main__":
    main()