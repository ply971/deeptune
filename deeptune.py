##### VI. GENERIC IMPORTS #####
from handlers.split_dataset import split_dataset
from pathlib import Path
from cli import DeepTuneVisionOptions
import pandas as pd 
from utils import RunType
from helpers import date_id,print_metrics_table,print_training_log_table, print_experiment_paths_table
from handlers.raw_to_parquet_dataset import raw_to_parquet
import json

defaults={
    'num_epochs':10,
    'learning_rate':1e-4,
    'added_layers':2,
    'embed_size':1000,
    'train_size':0.7,
    'val_size':0.1,
    'test_size':0.2,
    'mode':'cls',
    'fixed_seed':True,
    'disable_numerical_encoding':False,
    'gflu_stages':6,
    'target':'labels',
    'group_ids': None
}

def training_log_dir(ckpt_directory) -> Path:
    """Folder holding training_log.csv: the trainers write it next to the
    checkpoint (trainval_output_*/), so for a weights file that is its parent."""
    path = Path(ckpt_directory)
    return path.parent if path.is_file() else path


def main(argv=None):
    args = DeepTuneVisionOptions(RunType.ONECALL, args=argv)
    from desktop.config import RunConfig, validate_config, effective_mode
    values = {key: getattr(args, key) for key in RunConfig.__dataclass_fields__ if hasattr(args, key)}
    values["grouper"] = args.grouper or ""
    values["continuous_cols"] = args.continuous_cols or []
    values["categorical_cols"] = args.categorical_cols or []
    values["gandalf_type"] = args.type or "classification"
    values["added_layers"] = args.added_layers or 2
    values["embed_size"] = args.embed_size if args.embed_size is not None else 1000
    values["num_classes"] = args.num_classes if args.num_classes is not None else 2
    values["trainable_params"] = "" if args.trainable_params is None else str(args.trainable_params)
    cfg = RunConfig(**values)
    issues = validate_config(cfg)
    if args.trainable_params is not None and args.modality not in ('images', 'video'):
        issues.append('--trainable_params is only supported for the images and video modalities.')
    if args.modality == 'video' and getattr(args, 'video_flip', False):
        # A mirror swaps "moves left"/"moves right": refuse it on datasets that
        # declare direction-dependent labels (Phase 2 benchmarks' dataset.json).
        from benchmarks.guards import UnsafeAugmentationError, check_flip_augmentation
        try:
            check_flip_augmentation(args.df, hflip=True)
        except UnsafeAugmentationError as exc:
            issues.append(str(exc))
    # --- training basics (T1) ---
    from handlers.training_args import TrainingOptions
    training_options = TrainingOptions.from_args(args)
    issues.extend(training_options.issues(effective_mode(cfg)))
    if training_options.non_default() and args.modality not in ('images', 'video'):
        issues.append('These flags are only supported for the images and video modalities: '
                      + ', '.join('--' + name for name in training_options.non_default()))
    # --- end training basics ---
    # --- evaluation suite (B3) ---
    if args.modality != 'video':
        if args.validation != 'holdout':
            issues.append('--validation kfold and bootstrap632 are only supported for the video modality.')
        if args.export_dfanalyze:
            issues.append('--export_dfanalyze is only supported for the video modality.')
    elif args.validation != 'holdout' and training_options.resume is not None:
        issues.append('--resume cannot be combined with --validation: every fold must train a fresh model.')
    # --- end B3 ---
    # --- abstention (R2) ---
    if args.target_error is not None:
        if args.modality != 'video':
            issues.append('--target_error is only supported for the video modality.')
        elif effective_mode(cfg) != 'cls':
            issues.append('--target_error needs classification: a regression model has no error rate to bound.')
        elif args.calib_fraction >= defaults['train_size']:
            issues.append(f"--calib_fraction must be below the training share, {defaults['train_size']}.")
    # --- end R2 ---
    if issues:
        args.parser.error("\n".join(issues))
    args.mode = effective_mode(cfg)
    args.type = cfg.gandalf_type
    defaults = dict(globals()["defaults"], mode=args.mode, fixed_seed=args.fixed_seed, disable_numerical_encoding=args.mode == "reg")
    # --- training basics (T1): --seed varies training, not the test set; the split stays the fixed one ---
    if training_options.seed is not None:
        defaults['fixed_seed'] = True
    # --- end training basics ---

    parent_dir = date_id(root_dir=args.out)

    TARGET = args.target
    RAW_DATA = args.raw_data
    FINETUNING_MODE = args.finetuning_mode
    MODE = args.mode
    FREEZE_BACKBONE = args.freeze_backbone
    GROUPER = args.grouper

    # --num_epochs/--learning_rate are onecall-only args with their own CLI
    # defaults (10 / 1e-4, matching the values below) so they always override
    # cleanly. --added_layers/--embed_size are shared with other run types
    # and have no CLI-level default (None unless passed), so only override
    # when the user actually supplied one, preserving defaults otherwise.
    defaults['num_epochs'] = args.num_epochs
    defaults['learning_rate'] = args.learning_rate
    if args.added_layers is not None:
        defaults['added_layers'] = args.added_layers
    if args.embed_size is not None:
        defaults['embed_size'] = args.embed_size

    df_path = args.df if not RAW_DATA else raw_to_parquet(
        dataset_dir=args.df,
        out=Path(args.out)/parent_dir,
        modality=args.modality,
        video_storage=args.video_storage,
    )
    if args.modality == 'video':
        # A 'group' column (e.g. one source video per group) keeps related clips
        # in one split; recorded in args so embeddings carry the column too.
        from handlers.split_dataset import resolve_grouper
        import pyarrow.parquet as pq
        GROUPER = args.grouper = resolve_grouper(pq.read_schema(df_path).names, GROUPER, args.modality)

    train_data_path, val_data_path, test_data_path = split_dataset(
        train_size=defaults['train_size'],
        val_size=defaults['val_size'],
        test_size=defaults['test_size'],
        df_path=df_path,
        out_dir=Path(args.out)/parent_dir,
        fixed_seed=defaults['fixed_seed'],
        disable_numerical_encoding=defaults['disable_numerical_encoding'],
        target_column=TARGET,
        modality=args.modality,
        grouper=GROUPER,
        mode=args.mode,
        time_idx_column=args.time_idx_column,
    )

    # --- abstention (R2): whole groups move from training to a separate calibration split; validation and test do not change ---
    calibration_data_path = None
    if args.target_error is not None:
        from evaluators.video.abstention_fit import carve_calibration_split
        calibration_seed = training_options.seed if training_options.seed is not None else 42 if defaults['fixed_seed'] else None
        calibration_data_path = carve_calibration_split(train_data_path, args.calib_fraction, defaults['train_size'], GROUPER, calibration_seed)
    # --- end R2 ---

    df = pd.read_parquet(train_data_path, columns=['labels'])

    USE_CASE = 'peft' if args.use_peft else 'finetuned'

    MAPPING_PATH = next((p / "label_mapping.json" 
                     for p in Path(Path(args.out)/parent_dir).glob("data_splits*") 
                     if (p / "label_mapping.json").exists()), None)
    
    if args.modality in ('images', 'video', 'text') and args.mode == 'cls':
        mapping = json.loads(MAPPING_PATH.read_text(encoding='utf-8')) if MAPPING_PATH else {}
        classes = len(mapping) or int(df['labels'].nunique())
        if args.num_classes is None:
            args.num_classes = classes
        elif args.num_classes != classes:
            raise ValueError(f'Number of classes is {classes}, but --num_classes is {args.num_classes}.')

    if args.modality == 'text':
        from trainers.nlp.train_gpt2 import train as train_gpt2
        from trainers.nlp.train_multilingualbert import train as train_multilingualbert
        from evaluators.nlp.evaluate_multilingualbert import evaluate as evaluate_multilingualbert
        from evaluators.nlp.evaluate_gpt import evaluate as evaluate_gpt2
        from embed.nlp.gpt2_embeddings import embed as embed_gpt2
        from embed.nlp.multilingualbert_embeddings import embed as embed_multilingualbert
    
        if args.model_version == 'BERT':

            ckpt_directory = train_multilingualbert(
                out=Path(args.out)/parent_dir,
                batch_size=args.batch_size,
                train_df = train_data_path,
                val_df = val_data_path,
                num_epochs=defaults['num_epochs'],
                learning_rate=defaults['learning_rate'],
                added_layers=defaults['added_layers'],
                embed_size=defaults['embed_size'],
                freeze_backbone=FREEZE_BACKBONE,
                use_peft=args.use_peft,
                num_classes=args.num_classes,
                fixed_seed=defaults['fixed_seed'],
                model_str='peft-bert' if args.use_peft else 'bert',
                args=args

            )

            metrics_dict = evaluate_multilingualbert(
                eval_df=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                model_str='peft-bert' if args.use_peft else 'bert',
                num_classes=args.num_classes,
                added_layers=defaults['added_layers'],
                embed_size=defaults['embed_size'],
                batch_size=args.batch_size,
                use_peft=args.use_peft,
                args=args,
                freeze_backbone=FREEZE_BACKBONE,
            )


            exp_path,embed_shape = embed_multilingualbert(
                df_path=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                num_classes=args.num_classes,
                added_layers=defaults['added_layers'],
                embed_size=defaults['embed_size'],
                batch_size=args.batch_size,
                use_case=USE_CASE,
                freeze_backbone=FREEZE_BACKBONE,
            )


        elif args.model_version == 'gpt2':

            ckpt_directory = train_gpt2(
            out=Path(args.out)/parent_dir,
            batch_size=args.batch_size,
            train_df = train_data_path,
            val_df = val_data_path,
            num_epochs=defaults['num_epochs'],
            learning_rate=defaults['learning_rate'],
            freeze_backbone=FREEZE_BACKBONE,
            fixed_seed=defaults['fixed_seed'],
            use_peft=False,
            model_str='gpt2',
            args=args,
            num_classes=args.num_classes,
            )

            metrics_dict = evaluate_gpt2(
                eval_df=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                batch_size=args.batch_size,
                freeze_backbone=FREEZE_BACKBONE,
                args=args,
                use_peft=False,
                model_str='gpt2'
            )

            exp_path,embed_shape = embed_gpt2(
                df_path=test_data_path,
                out = Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                batch_size = args.batch_size,
                use_case="finetuned")
            
        csv_dir = training_log_dir(ckpt_directory)
        print_experiment_paths_table(df_path=df_path, train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path, ckpt_directory=ckpt_directory, exp_path=exp_path)
        print_training_log_table(csv_dir/"training_log.csv")
        print_metrics_table(metrics_dict, embed_shape, modality='text', mapping_path=MAPPING_PATH)
        
    elif args.modality == 'images':
        from trainers.vision.train import train as train_images
        from evaluators.vision.evaluate import evaluate as evaluate_images
        from embed.vision.embed import embed as embed_images

        ckpt_directory = train_images(
            train_df=train_data_path,
            val_df=val_data_path,
            out=Path(args.out)/parent_dir,
            batch_size=args.batch_size,
            num_epochs=defaults['num_epochs'],
            learning_rate=defaults['learning_rate'],
            added_layers=defaults['added_layers'],
            embed_size=defaults['embed_size'],
            freeze_backbone=FREEZE_BACKBONE,
            use_peft=args.use_peft,
            num_classes=args.num_classes,
            fixed_seed=defaults['fixed_seed'],
            args=args,
            mode=defaults['mode'],
            model_str=args.model_version,
            model_version=args.model_version,
            trainable_params=args.trainable_params,
        )
        if args.trainable_params is not None and args.embed_size is not None:
            # A PEFT budget picks the LoRA rank (= embed_size); eval/embed must rebuild that shape.
            defaults['embed_size'] = args.embed_size

        metrics_dict = evaluate_images(
            eval_df=test_data_path,
            mode=defaults['mode'],
            num_classes=args.num_classes,
            out=Path(args.out)/parent_dir,
            model_version=args.model_version,
            model_str=args.model_version,
            model_weights=ckpt_directory,
            use_peft=args.use_peft,
            added_layers=defaults['added_layers'],
            embed_size=defaults['embed_size'],
            batch_size=args.batch_size,
            freeze_backbone=FREEZE_BACKBONE,
            args=args
        )

        exp_path,embed_shape = embed_images(
            df_path=test_data_path,
            out=Path(args.out)/parent_dir,
            model_weights=ckpt_directory,
            batch_size=args.batch_size,
            use_case=USE_CASE,
            model_version=args.model_version,
            model_str=args.model_version,
            added_layers=defaults['added_layers'],
            embed_size=defaults['embed_size'],
            num_classes=args.num_classes,
            args=args,
            mode=defaults['mode'],
            grouper=GROUPER,
        )
        csv_dir = training_log_dir(ckpt_directory)
        print_experiment_paths_table(df_path=df_path, train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path, ckpt_directory=ckpt_directory, exp_path=exp_path)
        print_training_log_table(csv_dir/"training_log.csv")
        print_metrics_table(metrics_dict, embed_shape, modality='images',mapping_path=MAPPING_PATH)

    elif args.modality == 'video':
        from trainers.video.train import train as train_video
        from evaluators.video.evaluate import evaluate as evaluate_video
        from embed.video.embed import embed as embed_video

        # --- evaluation suite (B3) ---
        from evaluators.video.report import shared_source_warning
        video_split_warning = shared_source_warning(test_data_path, train_data_path, val_data_path)
        if video_split_warning:
            print(f'Warning: {video_split_warning}')
        # The folds run before the final model, and on train + validation rows only.
        video_eval_config = dict(mode=defaults['mode'], num_classes=args.num_classes, model_version=args.model_version, model_str=args.model_version,
                                 use_peft=args.use_peft, added_layers=defaults['added_layers'], embed_size=defaults['embed_size'],
                                 batch_size=args.batch_size, freeze_backbone=FREEZE_BACKBONE, num_frames=args.num_frames, pooling=args.pooling)
        if args.validation != 'holdout':
            from evaluators.video.run_validation import run_video_validation
            video_train_config = dict(batch_size=args.batch_size, num_epochs=defaults['num_epochs'], learning_rate=defaults['learning_rate'],
                                      added_layers=defaults['added_layers'], embed_size=defaults['embed_size'], freeze_backbone=FREEZE_BACKBONE,
                                      use_peft=args.use_peft, num_classes=args.num_classes, fixed_seed=defaults['fixed_seed'], mode=defaults['mode'],
                                      model_str=args.model_version, model_version=args.model_version, num_frames=args.num_frames,
                                      pooling=args.pooling, trainable_params=args.trainable_params)
            run_video_validation(args, train_data_path, val_data_path, Path(args.out)/parent_dir, train_video, evaluate_video,
                                 video_train_config, video_eval_config, GROUPER)
        # --- end B3 ---

        ckpt_directory = train_video(
            train_df=train_data_path,
            val_df=val_data_path,
            out=Path(args.out)/parent_dir,
            batch_size=args.batch_size,
            num_epochs=defaults['num_epochs'],
            learning_rate=defaults['learning_rate'],
            added_layers=defaults['added_layers'],
            embed_size=defaults['embed_size'],
            freeze_backbone=FREEZE_BACKBONE,
            use_peft=args.use_peft,
            num_classes=args.num_classes,
            fixed_seed=defaults['fixed_seed'],
            args=args,
            mode=defaults['mode'],
            model_str=args.model_version,
            model_version=args.model_version,
            num_frames=args.num_frames,
            pooling=args.pooling,
            trainable_params=args.trainable_params,
        )
        if args.trainable_params is not None and args.embed_size is not None:
            # A PEFT budget on a 2D backbone picks the LoRA rank (= embed_size); eval/embed must rebuild
            # that shape. Native models keep embed_size, and eval/embed read args.lora_rank.
            defaults['embed_size'] = args.embed_size

        metrics_dict = evaluate_video(
            eval_df=test_data_path,
            mode=defaults['mode'],
            num_classes=args.num_classes,
            out=Path(args.out)/parent_dir,
            model_version=args.model_version,
            model_str=args.model_version,
            model_weights=ckpt_directory,
            use_peft=args.use_peft,
            added_layers=defaults['added_layers'],
            embed_size=defaults['embed_size'],
            batch_size=args.batch_size,
            freeze_backbone=FREEZE_BACKBONE,
            args=args,
            num_frames=args.num_frames,
            pooling=args.pooling,
        )

        exp_path,embed_shape = embed_video(
            df_path=test_data_path,
            out=Path(args.out)/parent_dir,
            model_weights=ckpt_directory,
            batch_size=args.batch_size,
            use_case=USE_CASE,
            model_version=args.model_version,
            model_str=args.model_version,
            added_layers=defaults['added_layers'],
            embed_size=defaults['embed_size'],
            num_classes=args.num_classes,
            args=args,
            mode=defaults['mode'],
            grouper=GROUPER,
            num_frames=args.num_frames,
            pooling=args.pooling,
        )
        csv_dir = training_log_dir(ckpt_directory)
        print_experiment_paths_table(df_path=df_path, train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path, ckpt_directory=ckpt_directory, exp_path=exp_path)
        print_training_log_table(csv_dir/"training_log.csv")
        print_metrics_table(metrics_dict, embed_shape, modality='video',mapping_path=MAPPING_PATH)
        # --- evaluation suite (B3): last, so that a problem here cannot cost the results above ---
        if args.export_dfanalyze:
            from embed.video.dfanalyze import export_from_args
            export_from_args(args, {'train': train_data_path, 'val': val_data_path, 'test': test_data_path}, ckpt_directory, USE_CASE,
                             args.model_version, defaults['added_layers'], defaults['embed_size'], args.num_classes, defaults['mode'],
                             Path(args.out)/parent_dir, args.model_version, args.batch_size, args.num_frames, args.pooling, GROUPER)
        if args.report_boot > 0:
            from evaluators.video.run_validation import write_final_report
            write_final_report(args, Path(args.out)/parent_dir, ckpt_directory, test_data_path, val_data_path, train_data_path, MAPPING_PATH,
                               evaluate_video, dict(video_eval_config, embed_size=defaults['embed_size']), GROUPER,
                               extra_warnings=[video_split_warning] if video_split_warning else [])
        # --- end B3 ---
        # --- abstention (R2): after the test scores, which it does not see; the report then checks the fitted rule on them ---
        if args.target_error is not None:
            from datasets.video_datasets import video_options_from_args
            from evaluators.video.abstention_fit import fit_on_calibration_split, report_on_eval_dir
            from utils import is_native_video_architecture, resolve_lora_rank
            abstention_rule = fit_on_calibration_split(
                calibration_data_path, ckpt_directory, target_error=args.target_error, delta=args.abstain_delta, guarantee=args.abstain_guarantee,
                score_name=args.abstain_score, model_version=args.model_version, num_classes=args.num_classes, added_layers=defaults['added_layers'],
                embed_size=defaults['embed_size'], freeze_backbone=FREEZE_BACKBONE, use_peft=args.use_peft, batch_size=args.batch_size, num_frames=args.num_frames,
                pooling=args.pooling, video_options=video_options_from_args(args), grouper=GROUPER, other_splits=(train_data_path, val_data_path, test_data_path),
                lora_rank=resolve_lora_rank(args, ckpt_directory) if args.use_peft and is_native_video_architecture(args.model_version) else None)
            report_on_eval_dir(abstention_rule, next(path for path in sorted((Path(args.out)/parent_dir).glob('eval_output_*')) if path.is_dir()), test_data_path, GROUPER,
                               n_boot=args.report_boot, calib_df=calibration_data_path)
        # --- end R2 ---

    elif args.modality == 'tabular':

        if args.model_version == 'gandalf':
            from trainers.tabular.train_gandalf import train as train_tabular_gandalf
            from evaluators.tabular.evaluate_gandalf import evaluate as evaluate_tabular_gandalf
            from embed.tabular.gandalf_embeddings import embed as embed_tabular_gandalf

            ckpt_directory = train_tabular_gandalf(
                train_df=train_data_path,
                val_df=val_data_path,
                out=Path(args.out)/parent_dir,
                batch_size=args.batch_size,
                num_epochs=defaults['num_epochs'],
                learning_rate=defaults['learning_rate'],
                gflu_stages=defaults['gflu_stages'],
                target=[defaults['target']],
                continuous_cols=args.continuous_cols,
                categorical_cols=args.categorical_cols,
                model_str='GANDALF',
                args=args,
                type=args.type
            )
            metrics_dict = evaluate_tabular_gandalf(
                eval_df=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                model_str='GANDALF',
                args=args,
            )


            exp_path,embed_shape = embed_tabular_gandalf(
                eval_df=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                cont_cols=args.continuous_cols,
                cat_cols=args.categorical_cols,
                batch_size=args.batch_size,
                target=defaults['target'],
                args=args,
                grouper = GROUPER,
                model_str='GANDALF',
            )
            print_experiment_paths_table(df_path=df_path, train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path, ckpt_directory=ckpt_directory, exp_path=exp_path)
            print_metrics_table(metrics_dict, embed_shape, modality='tabular',mapping_path=MAPPING_PATH)

        if args.model_version == 'tabpfn':
            ### For TabPFN ###
            train_df = pd.read_parquet(train_data_path)
            X_train_tabpfn = train_df.drop(columns=['labels'])
            y_train_tabpfn = train_df['labels']
            val_df = pd.read_parquet(val_data_path)
            X_val_tabpfn = val_df.drop(columns=['labels'])
            y_val_tabpfn = val_df['labels']
            eval_df = pd.read_parquet(test_data_path)
            X_eval_tabpfn = eval_df.drop(columns=['labels'])
            y_eval_tabpfn = eval_df['labels']


            from trainers.tabular.train_tabpfn import train_tabpfn_from_scratch as train_tabular_tabpfn
            from trainers.tabular.train_tabpfn import finetune_tabpfn as finetune_tabular_tabpfn
            from evaluators.tabular.evaluate_tabpfn import evaluate_tabpfn as evaluate_tabular_tabpfn
            from embed.tabular.tabpfn_embeddings import get_tabpfn_embeddings as embed_tabpfn_embeddings

            if FINETUNING_MODE:

                ckpt_directory = finetune_tabular_tabpfn(
                    X_train = X_train_tabpfn,
                    y_train = y_train_tabpfn,
                    X_val = X_val_tabpfn,
                    y_val = y_val_tabpfn,
                    mode = MODE,
                    out = Path(args.out)/parent_dir,
                    num_epochs=defaults['num_epochs'],
                    args=args,
                    model_str='TABPFN',
                )


            else:
                ckpt_directory = train_tabular_tabpfn(
                    X_train= X_train_tabpfn,
                    y_train= y_train_tabpfn,
                    mode=MODE,
                    out=Path(args.out)/parent_dir,
                    X_val= X_val_tabpfn,
                    y_val= y_val_tabpfn,
                    args=args,
                    model_str='TABPFN',
                )

            metrics_dict = evaluate_tabular_tabpfn(
                X_eval_tabpfn,
                y_eval_tabpfn,
                out=Path(args.out)/parent_dir,
                model_path=ckpt_directory,
                mode=MODE,
                args=args,
                finetuning_mode=FINETUNING_MODE,
                model_str='TABPFN',
                )
            
            exp_path,embed_shape = embed_tabpfn_embeddings(
                X_train_tabpfn,
                y_train_tabpfn,
                X_eval_tabpfn,
                y_eval_tabpfn,
                out=Path(args.out)/parent_dir,
                args=args,
                model_path=ckpt_directory,
                mode=MODE,
                finetuning_mode=FINETUNING_MODE,
                grouper=GROUPER,
                model_str='TABPFN',
            )

            print_experiment_paths_table(df_path=df_path, train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path, ckpt_directory=ckpt_directory, exp_path=exp_path)
            print_metrics_table(metrics_dict, embed_shape, modality='tabular')

    elif args.modality == 'timeseries':
        from trainers.timeseries.train_deepar import train as train_deepar
        from evaluators.timeseries.evaluate_deepar import evaluate as evaluate_deepar
        from embed.timeseries.deepAR_embeddings import embed as embed_deepar
        if args.model_version == 'deepAR':

            ckpt_directory = train_deepar(
                train_df=train_data_path,
                val_df=val_data_path,
                out=Path(args.out)/parent_dir,
                batch_size=args.batch_size,
                num_epochs=defaults['num_epochs'],
                learning_rate=defaults['learning_rate'],
                timeindex_column=args.time_idx_column,
                target_column=defaults['target'],
                group_ids=[GROUPER] if GROUPER else defaults['group_ids'],
                args=args,
                model_str='DeepAR',
            )


            exp_path,_ = evaluate_deepar(
                train_df_path=train_data_path,
                val_df_path=val_data_path,
                eval_df_path=test_data_path,
                out=Path(args.out)/parent_dir,
                batch_size=args.batch_size,
                timeindex_column=args.time_idx_column,
                target_column=defaults['target'],
                model_weights=ckpt_directory,
                group_ids=[GROUPER] if GROUPER else defaults['group_ids'],
                args=args,
            )

            exp_path, embed_shape = embed_deepar(
                eval_df=test_data_path,
                out=Path(args.out)/parent_dir,
                model_weights=ckpt_directory,
                batch_size=args.batch_size,
                target_column=defaults['target'],
                args=args,
                history_df_paths=[train_data_path, val_data_path],
                group_ids=[GROUPER] if GROUPER else None,
                model_str='DeepAR',
                timeindex_column=args.time_idx_column,
            )

            print_experiment_paths_table(df_path=df_path, train_data_path=train_data_path, val_data_path=val_data_path, test_data_path=test_data_path, ckpt_directory=ckpt_directory, exp_path=exp_path)

    print(" ✅ Your run is complete. Check the output directory table for a detailed copy of your results. \n Thank you for using DeepTune!🚀")


        

    

if __name__ == "__main__":
    main()






    

