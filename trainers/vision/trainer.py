from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sys
import logging
from helpers import PerformanceLogger
from options import DEVICE, PIN_MEM, TRAINVAL_OUTPUT_DIR
import time
from utils import save_process_times
import functools
import contextlib
import math
import os
from handlers.training_args import TrainingOptions
from trainers.training_basics import (
    FREEZE_BN_AUTO_BELOW, PEAK_MEMORY_MEANING, ExperimentLogger, ModelEma, autocast_context, backbone_batchnorm_layers,
    build_scheduler, decide_freeze_bn, get_rng_state, install_worker_seeding,
    layer_wise_param_groups, mix_batch, peak_memory_mb, reset_peak_memory, resolve_precision, run_environment,
    run_peak_memory, set_rng_state, trajectory_differences,
)
print = functools.partial(print, flush=True)

CHECKPOINT_FILENAME = 'last_checkpoint.pt'
BEST_WEIGHTS_FILENAME = 'best_model_weights.pth'
RUN_RECORD_FILENAME = 'training_run.json'
CHECKPOINT_FORMAT_VERSION = 1


def _atomic_save(obj, path) -> None:
    """A crash while writing must not destroy the previous good file."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def _dataset_size(loader):
    try:
        return len(loader.dataset)
    except (AttributeError, TypeError):
        return None


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        mode: str,
        num_epochs: int,
        output_dir: Path = TRAINVAL_OUTPUT_DIR,
        training_options: TrainingOptions | None = None,
        backbone_pretrained: bool = False,
    ):
        """
        Performs Training & Validation on the input image dataset.
        
        Args:
        
            model (PyTorch Model): The model we are loading from the src file, whether it is for transfer learning with PEFT Or without.
            train_loader (torch.utils.data.DataLoader): The DataLoader for the training set.
            val_loader (torch.utils.data.DataLoader): The DataLoader for the validation set.
            learning_rate (float): The learning rate for the optimizer.
            mode (str): The mode of the model, either classification or regression.
            num_epochs (int): The number of epochs to train the model.
            output_dir (str): The directory to save the training logs and model checkpoints.
            training_options (TrainingOptions): Mixed precision, resume, early stopping, schedules and the rest of the
                training-basics flags (handlers/training_args.py). None means the defaults, i.e. a plain Adam loop.
            backbone_pretrained (bool): Whether the backbone holds pretrained weights. --freeze_bn auto stays off without them.

        Attributes:
        
            criterion (torch.nn.Module): Loss function, Cross Entropy as we do classification.
            performance_logger (PerformanceLogger): Logger instance for tracking training and validation progress.
            logger (logging.Logger): Logger instance for tracking training progress.
        """
        
        self.model = model
        self.model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mode = mode
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        if mode not in ('cls', 'reg') or num_epochs < 1:
            raise ValueError('Choose cls or reg and at least one training epoch.')
        if len(train_loader) == 0 or len(val_loader) == 0:
            raise ValueError('Training and validation datasets must both contain samples.')
        
        if self.mode == 'cls':    
            self.criterion = nn.CrossEntropyLoss()
        else: # then regression if not classification
            self.criterion = nn.MSELoss()
        self.options = (training_options or TrainingOptions()).validate(mode)
        # Before the optimizer is built, so that it never sees the frozen BN parameters.
        bn_layers = backbone_batchnorm_layers(self.model)
        physical_batch = getattr(train_loader, 'batch_size', None)
        freeze_bn, bn_reason = decide_freeze_bn(self.options.freeze_bn, physical_batch, len(bn_layers), backbone_pretrained)
        self.frozen_bn_layers = bn_layers if freeze_bn else []
        for layer in self.frozen_bn_layers:
            layer.requires_grad_(False)
        self.freeze_bn_info = {'requested': self.options.freeze_bn, 'active': freeze_bn, 'reason': bn_reason,
                               'frozen_layers': len(self.frozen_bn_layers), 'physical_batch': physical_batch,
                               'backbone_pretrained': bool(backbone_pretrained), 'auto_cutoff': FREEZE_BN_AUTO_BELOW}
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if not params:
            raise ValueError("No trainable parameters found in the model!")
        if self.options.layer_decay < 1:
            self.optimizer = optim.Adam(layer_wise_param_groups(model, self.learning_rate, self.options.layer_decay),
                                        lr=self.learning_rate)
        else:
            self.optimizer = optim.Adam(params, lr=self.learning_rate)

        self.performance_logger = PerformanceLogger(output_dir) if output_dir else None

        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()

        # --- trainable / total parameter report (A1 param budget) ---
        import json
        from trainers.param_budget import summarize_parameters, format_summary, format_lora_coverage, lora_size_warning
        self.parameter_summary = summarize_parameters(self.model)
        self.logger.info(format_summary(self.parameter_summary))
        lora_line = format_lora_coverage(self.parameter_summary)
        if lora_line:
            self.logger.info(lora_line)
        lora_warning = lora_size_warning(self.parameter_summary)
        if lora_warning:
            self.logger.warning(lora_warning)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(output_dir) / 'parameter_summary.json', 'w', encoding='utf-8') as stream:
                json.dump(self.parameter_summary, stream, indent=2)
        # --- end parameter report ---

        # --- training basics (T1) ---
        options = self.options
        if options.freeze_bn != 'off':
            self.logger.info(bn_reason)
        if freeze_bn and not backbone_pretrained:
            self.logger.warning(bn_reason)
        if options.grad_accum_steps > 1 and not freeze_bn and any(isinstance(m, _BatchNorm) for m in self.model.modules()):
            self.logger.warning(f'--grad_accum_steps {options.grad_accum_steps} raises the batch size for the gradients only. '
                                f'BatchNorm still computes its statistics from each batch of {physical_batch}; '
                                f'see --freeze_bn.')
        # The smoothed loss trains; self.criterion stays the plain one that
        # validation reports and selects on, so scores remain comparable.
        self.train_criterion = nn.CrossEntropyLoss(label_smoothing=options.label_smoothing) if options.label_smoothing > 0 else None
        self.amp_dtype, use_scaler, precision_line = resolve_precision(options.precision, DEVICE)
        if precision_line:
            self.logger.info(precision_line)
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
        self.updates_per_epoch = math.ceil(len(train_loader) / options.grad_accum_steps)
        self.scheduler = build_scheduler(self.optimizer, options, num_epochs, self.updates_per_epoch, learning_rate)
        self.ema = ModelEma(self.model, options.ema_decay) if options.ema_decay > 0 else None
        if options.layer_decay < 1:
            rates = [group['lr'] for group in self.optimizer.param_groups]
            self.logger.info(f'Layer-wise LR decay {options.layer_decay}: {len(rates)} groups, lr {min(rates):.3g} to {max(rates):.3g}.')
        if options.seed is not None:
            install_worker_seeding(train_loader)
            install_worker_seeding(val_loader)
        self.seed_effective = options.seed if options.seed is not None else torch.initial_seed()
        # Identity of the data this run trains on; trainers/*/train.py add file
        # fingerprints. A resume on other data would mix splits, see _load_checkpoint.
        self.data_identity = {'train_samples': _dataset_size(train_loader), 'val_samples': _dataset_size(val_loader)}
        self.completed_epochs = 0
        self.global_updates = 0
        self.best_metric = None
        self.best_epoch = None
        self.epochs_without_improvement = 0
        self.stopped_early = False
        self.resumed_from = None
        self.epoch_times = []
        self.total_time = 0.0
        self.experiment = None
        # --- end training basics ---

    # --- training basics (T1) ---
    @property
    def _best_weights_path(self):
        return Path(self.output_dir) / BEST_WEIGHTS_FILENAME if self.output_dir else None

    @property
    def _checkpoint_path(self):
        return Path(self.output_dir) / CHECKPOINT_FILENAME if self.output_dir else None

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[-1]['lr']  # the head's rate when --layer_decay is on

    def _training_loss(self, outputs, labels, other_labels, lam):
        criterion = self.train_criterion or self.criterion
        if other_labels is None:
            return criterion(outputs, labels)
        return lam * criterion(outputs, labels) + (1.0 - lam) * criterion(outputs, other_labels)

    def _optimizer_update(self, window_samples: int) -> None:
        """Apply one accumulation window. With accumulation the gradients are
        sums over samples, so dividing by the window's sample count gives the
        mean over the window for any mix of batch sizes."""
        if self.options.grad_accum_steps > 1:
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    if param.grad is not None:
                        param.grad.div_(window_samples)
        scale_before = self.scaler.get_scale() if self.scaler.is_enabled() else None
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if scale_before is not None and self.scaler.get_scale() < scale_before:
            return  # GradScaler skipped this step (inf/NaN gradients): the weights did not move
        self.global_updates += 1
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model)

    def _validate_epoch(self):
        """Validation scores for this epoch, on the EMA weights when EMA is on.
        Also updates the best-so-far, which must see the same weights."""
        with (self.ema.swapped_in(self.model) if self.ema is not None else contextlib.nullcontext()):
            result = self.validate()
            val_loss, val_accuracy = result if self.mode == 'cls' else (result, None)
            improved = self._track_best(val_loss, val_accuracy)
        return val_loss, val_accuracy, improved

    def _track_best(self, val_loss, val_accuracy) -> bool:
        options = self.options
        if not options.selects_best:
            return False
        value = val_accuracy if options.selection_metric == 'val_accuracy' else val_loss
        better = (self.best_metric is None or
                  (value > self.best_metric + options.early_stopping_min_delta if options.selection_metric == 'val_accuracy'
                   else value < self.best_metric - options.early_stopping_min_delta))
        if not math.isfinite(value) or not better:
            self.epochs_without_improvement += 1
            return False
        self.best_metric = float(value)
        self.best_epoch = self.completed_epochs + 1
        self.epochs_without_improvement = 0
        if self._best_weights_path is not None:
            _atomic_save(self.model.state_dict(), self._best_weights_path)
        else:
            self._best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        return True

    def _save_checkpoint(self) -> None:
        best_path = self._best_weights_path
        payload = {
            'format_version': CHECKPOINT_FORMAT_VERSION,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'scaler': self.scaler.state_dict() if self.scaler.is_enabled() else None,
            'ema': self.ema.state_dict() if self.ema is not None else None,
            'epoch': self.completed_epochs,
            'global_updates': self.global_updates,
            'rng': get_rng_state(),
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'epochs_without_improvement': self.epochs_without_improvement,
            'stopped_early': self.stopped_early,
            # Stored inside, so the checkpoint and its best-so-far can never disagree.
            'best_weights': (torch.load(best_path, map_location='cpu', weights_only=True)
                             if self.best_epoch is not None and best_path.is_file() else None),
            'options': self.options.to_dict(),
            'seed_effective': self.seed_effective,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'batch_size': getattr(self.train_loader, 'batch_size', None),
            'data_identity': self.data_identity,
            'performance_log': self.performance_logger.log_data if self.performance_logger else None,
            'metric_rows': self.experiment.rows if self.experiment else [],
            'epoch_times': self.epoch_times,
            'total_time': self.total_time,
        }
        _atomic_save(payload, self._checkpoint_path)
        self.logger.info(f'Checkpoint written after epoch {self.completed_epochs}: {self._checkpoint_path}')

    def _load_checkpoint(self, path) -> None:
        """Restore the whole training state, so that the next epoch continues
        as if the run had never stopped. RNG last: nothing may draw after it."""
        try:
            state = torch.load(path, map_location='cpu', weights_only=True)
        except Exception as exc:
            raise ValueError(f'--resume: could not read {path} as a DeepTune checkpoint: {exc}') from exc
        if not isinstance(state, dict) or 'format_version' not in state:
            raise ValueError(f'--resume: {path} is not a DeepTune training checkpoint. model_weights.pth holds '
                             f'weights only; use the last_checkpoint.pt written with --checkpoint_every.')
        if state['format_version'] != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(f"--resume: {path} has checkpoint format {state['format_version']}, "
                             f'this DeepTune reads format {CHECKPOINT_FORMAT_VERSION}.')
        self._check_resume_data(state)
        if state['best_epoch'] is not None and not self.options.selects_best:
            raise ValueError(f"--resume: {path} selected its best weights on validation (best epoch {state['best_epoch']}"
                             f"{', stopped early' if state['stopped_early'] else ''}), but this run does not select. It "
                             f'would hand on the last weights and record that validation chose nothing. Pass --select_best '
                             f'or the original --early_stopping_patience.')
        saved_setup = dict(state['options'], learning_rate=state['learning_rate'], num_epochs=state['num_epochs'],
                           batch_size=state['batch_size'])
        current_setup = dict(self.options.to_dict(), learning_rate=self.learning_rate, num_epochs=self.num_epochs,
                             batch_size=getattr(self.train_loader, 'batch_size', None))
        differences = trajectory_differences(saved_setup, current_setup)
        if differences:
            self.logger.warning('--resume: these settings differ from the checkpoint, so this will not match an '
                                'uninterrupted run: ' + '; '.join(differences))
        try:
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            if self.ema is not None:
                if state['ema'] is None:
                    raise ValueError('the checkpoint has no EMA state, but --ema_decay is on')
                self.ema.load_state_dict(state['ema'])
        except (RuntimeError, ValueError, KeyError) as exc:
            raise ValueError(f'--resume: {path} does not fit this model or its settings ({exc}). Use the model flags '
                             f'of the original run.') from exc
        if self.scheduler is not None and state['scheduler'] is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        if self.scaler.is_enabled() and state['scaler'] is not None:
            self.scaler.load_state_dict(state['scaler'])
        self.completed_epochs = state['epoch']
        self.global_updates = state['global_updates']
        self.best_metric = state['best_metric']
        self.best_epoch = state['best_epoch']
        self.epochs_without_improvement = state['epochs_without_improvement']
        self.stopped_early = state['stopped_early']
        self.epoch_times = list(state['epoch_times'])
        self.total_time = state['total_time']
        self.experiment.rows = list(state['metric_rows'])
        self.seed_effective = state['seed_effective']
        if self.performance_logger and state['performance_log'] is not None:
            self.performance_logger.log_data = state['performance_log']
        if state['best_weights'] is not None:
            if self._best_weights_path is not None:
                _atomic_save(state['best_weights'], self._best_weights_path)
            else:
                self._best_state = state['best_weights']
        self.resumed_from = str(path)
        set_rng_state(state['rng'], warn=self.logger.warning)
        self.logger.info(f'Resumed from {path}: {self.completed_epochs} epochs done, {self.global_updates} updates.')

    def _check_resume_data(self, state) -> None:
        saved = state['data_identity']
        different = [key for key in sorted(set(saved) | set(self.data_identity))
                     if saved.get(key) is not None and self.data_identity.get(key) is not None
                     and saved[key] != self.data_identity[key]]
        if not different:
            return
        message = (f"--resume: the data differ from the checkpoint's ({', '.join(different)}). Continuing would put "
                   f'samples the model already trained on into validation or test. With deeptune.py, use --fixed-seed '
                   f'or --seed in both runs so that the split is the same.')
        if not self.options.resume_allow_data_change:
            raise ValueError(message + ' Pass --resume_allow_data_change for a deliberate warm start.')
        self.logger.warning(message + ' Going on because of --resume_allow_data_change.')

    def _reload_final_weights(self) -> None:
        """Leave the model holding the weights this run hands on: the best ones
        when selecting, else the EMA, else the last. Runs before the error-model
        fit, so that it describes the weights train.py saves (V-011)."""
        if self.options.selects_best and self.best_epoch is not None:
            path = self._best_weights_path
            best = (torch.load(path, map_location='cpu', weights_only=True) if path is not None
                    else self._best_state)
            self.model.load_state_dict(best)
            self.logger.info(f'Reloaded the best weights: epoch {self.best_epoch}, '
                             f'{self.options.selection_metric} = {self.best_metric:.5f}.')
        elif self.ema is not None:
            self.ema.copy_to(self.model)
            self.logger.info('The model now holds the EMA weights.')

    def selection_info(self) -> dict:
        """What the validation split was used for; R2's conformal calibration
        needs to know, because a selection set is no longer exchangeable with test."""
        options = self.options
        return {
            'selection_set': 'validation' if options.selects_best else None,
            'validation_used_for_model_selection': bool(options.selects_best),
            'selection_metric': options.selection_metric if options.selects_best else None,
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'epochs_completed': self.completed_epochs,
            'stopped_early': self.stopped_early,
            'early_stopping_patience': options.early_stopping_patience,
        }

    def _write_run_record(self) -> None:
        if not self.output_dir:
            return
        import json
        record = {
            'seed_requested': self.options.seed,
            'seed_effective': self.seed_effective,
            'options': self.options.to_dict(),
            'precision_active': str(self.amp_dtype).replace('torch.', '') if self.amp_dtype is not None else 'float32',
            'grad_scaler': self.scaler.is_enabled(),
            'optimizer_updates': self.global_updates,
            'updates_per_epoch': self.updates_per_epoch,
            'final_lr': self._current_lr(),
            'selection': self.selection_info(),
            'freeze_bn': self.freeze_bn_info,
            'resumed_from': self.resumed_from,
            'data_identity': self.data_identity,
            'peak_memory': run_peak_memory(self.experiment.rows),
            'peak_memory_meaning': PEAK_MEMORY_MEANING,
            'environment': run_environment(),
        }
        (Path(self.output_dir) / RUN_RECORD_FILENAME).write_text(json.dumps(record, indent=2), encoding='utf-8')
    # --- end training basics ---

    def train(self):

        # --- training basics (T1) ---
        options = self.options
        self.experiment = ExperimentLogger(
            options.experiment_logger, self.output_dir, run_name=options.experiment_name,
            params={**options.to_dict(), 'learning_rate': self.learning_rate, 'num_epochs': self.num_epochs,
                    'seed_effective': self.seed_effective, 'mode': self.mode},
            warn=self.logger.warning)
        if options.resume:
            self._load_checkpoint(options.resume)
        start_epoch = self.completed_epochs
        patience = options.early_stopping_patience
        if patience > 0 and self.epochs_without_improvement >= patience:
            self.stopped_early = True
        accum = options.grad_accum_steps
        n_batches = len(self.train_loader)
        # --- end training basics ---
        total_time = self.total_time
        epoch_times = self.epoch_times
        for epoch in range(start_epoch, self.num_epochs):
            if self.stopped_early:
                break

            self.model.train()
            for layer in self.frozen_bn_layers:  # model.train() switched them back on
                layer.eval()
            epoch_start = time.time()
            reset_peak_memory(DEVICE)
            window_samples = 0

            
            """
            The following arguments are as follows:
            
            - running_loss: indicating the loss during the epoch for train and val datasets respectively.
            - correct_predictions: number of predictions that equal true label 
            - total_predictions: number of predictions made in total
            """

            running_loss=0.0
            samples_seen = 0
            correct_predictions=0.0
            total_predictions=0
            train_pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))

            for i, (inputs, labels,*_) in train_pbar:
                
                # make sure the inputs and labels on GPU
                
                if isinstance(self.criterion, nn.MSELoss): # if regression then we must rehape the target tensor    
                    inputs, labels = inputs.to(DEVICE, non_blocking=PIN_MEM), labels.to(DEVICE, non_blocking=PIN_MEM)
                    labels = labels.view(-1,1).float() # [batch_size,1]
                else:
                    inputs,labels = inputs.to(DEVICE, non_blocking=PIN_MEM),labels.to(DEVICE, non_blocking=PIN_MEM)
                    

                # null the gradient at the start of each accumulation window
                if i % accum == 0:
                    self.optimizer.zero_grad()
                    window_samples = 0

                inputs, other_labels, lam = mix_batch(inputs, labels, options.mixup_alpha, options.cutmix_alpha, options.mix_prob)

                # run the model and compute the loss
                with autocast_context(self.amp_dtype):
                    outputs = self.model(inputs)
                    loss = self._training_loss(outputs, labels, other_labels, lam)

                # backprop; with accumulation the gradients add up as sums over samples
                self.scaler.scale(loss if accum == 1 else loss * labels.size(0)).backward()
                window_samples += labels.size(0)

                # apply the optimizer at the end of a window, including a shorter last one
                if (i + 1) % accum == 0 or i + 1 == n_batches:
                    self._optimizer_update(window_samples)

                # accumulate loss
                running_loss += loss.item() * labels.size(0)
                samples_seen += labels.size(0)

                if self.mode == "cls":

                    # calculate accuracy; a mixed sample counts for both of its labels by their weights
                    predicted = torch.argmax(outputs,dim=1)
                    correct_predictions += torch.sum(predicted==labels).item() if other_labels is None else (
                        lam * torch.sum(predicted==labels).item() + (1.0 - lam) * torch.sum(predicted==other_labels).item())
                    total_predictions += labels.size(0)

                    # now we compute the average loss and accuracy for each epoch
                    epoch_accuracy = 100. * correct_predictions / total_predictions
                    
                # update tqdm progress bar
                train_pbar.set_postfix({"loss": round(running_loss / samples_seen, 5)})
            # update training loss
            epoch_loss = running_loss / samples_seen
            
            epoch_end = time.time()
            epoch_duration = epoch_end - epoch_start
            total_time += epoch_duration

            # record the time taken for the current epoch
            epoch_times.append({"epoch": epoch + 1, "duration_seconds": epoch_duration})
        
            if self.mode == 'cls':
                
                """
                Here in Classification we have both loss and accuracy as metrics to track, so we will log both of them.
                """
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_loss:.3f}, Training Accuracy: {epoch_accuracy:.3f} % "
                )    
                # Then we see how validation set works
                val_loss, val_accuracy, improved = self._validate_epoch()
                self.logger.info(f"Validation loss: {val_loss:.3f} , Valiadation Accuracy: {val_accuracy:.3f} % ")
            
            else:
                """
                Only loss is needed to be logged in regression mode.
                """
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_loss}"
                )    
    
                val_loss, val_accuracy, improved = self._validate_epoch()

            if self.performance_logger:

                if self.mode == 'cls':
                    
                    self.performance_logger.log_epoch(
                        epoch = epoch+1,
                        epoch_loss=epoch_loss,
                        epoch_accuracy=epoch_accuracy,
                        val_loss=val_loss,
                        val_accuracy=val_accuracy,
                        
                    )
            
                else:
                    
                    self.performance_logger.log_epoch(
                        epoch = epoch+1,
                        epoch_loss=epoch_loss,
                        val_loss=val_loss,
                        epoch_accuracy='Regression: No Accuracy',
                        val_accuracy = 'Regression: No Accuracy'
                        
                    )    
                
                self.performance_logger.save_to_csv(f"{self.output_dir}/training_log.csv")

            # --- training basics (T1) ---
            self.completed_epochs = epoch + 1
            self.total_time = total_time
            self.experiment.log_epoch({
                'epoch': epoch + 1, 'train_loss': epoch_loss, 'train_accuracy': epoch_accuracy if self.mode == 'cls' else None,
                'val_loss': val_loss, 'val_accuracy': val_accuracy, 'lr': self._current_lr(),
                'optimizer_updates': self.global_updates, 'seconds': epoch_duration, 'is_best': improved,
                **peak_memory_mb(DEVICE),
            })
            if patience > 0 and self.epochs_without_improvement >= patience:
                self.stopped_early = True
                self.logger.info(f'Early stopping after epoch {epoch + 1}: no improvement in {options.selection_metric} '
                                 f'for {patience} epochs (best: epoch {self.best_epoch}).')
            if options.checkpoint_every > 0 and self.output_dir and (
                    self.completed_epochs % options.checkpoint_every == 0
                    or self.completed_epochs == self.num_epochs or self.stopped_early):
                self._save_checkpoint()
            # --- end training basics ---

        # record the total training time at the end
        save_process_times(epoch_times, total_time, self.output_dir,"training")

        # --- training basics (T1): hand on the selected weights and say what selected them ---
        self._reload_final_weights()
        if options.selects_best:
            self.logger.warning('The validation split was used to select these weights, so scores on it are optimistic. '
                                'See training_run.json; calibration that needs exchangeable data must use another split.')
        self.experiment.finish()
        self._write_run_record()
        # --- end training basics ---

        # --- per-prediction error model (A4) ---
        # One extra pass over the VALIDATION loader (never test data) with the
        # final weights -- the ones train.py saves as model_weights.pth -- fits
        # error_model.json next to them. On by default; trainers/*/train.py
        # set trainer.fit_error_model = not args.no_error_model. RNG state and
        # train/eval mode are restored, so training results are unchanged.
        # With --select_best / early stopping the best weights were reloaded
        # above, so the fit sees them (V-011), but validation is then a
        # selection set: fit_info records that, and a separate calibration
        # split is R2's job (P-104). The weights fingerprint recorded in
        # error_model_fit.json makes eval / inference refuse an error model
        # fitted on other weights.
        if getattr(self, 'fit_error_model', True) and self.output_dir:
            try:
                self._fit_error_model()
            except Exception as exc:  # never let the auxiliary estimate break a finished training run
                self.logger.warning(f'Per-prediction error model was not fitted: {exc}')
        # --- end per-prediction error model (A4) ---

    # --- per-prediction error model (A4) ---
    def _fit_error_model(self):
        """Fit the per-prediction error model on the validation split and
        save ``error_model.json`` + ``error_model_fit.json`` in output_dir.
        Returns the fitted ``evaluators.error_model.ErrorModel``.

        Optional hooks (set by the caller, like ``fit_error_model``):
        ``error_fit_loader`` / ``error_fit_model`` replace the validation loader /
        model for this fit (video ``--test_clips``: N clips per video through
        ``src.video.multi_clip.MultiClipModel``, so the fit sees the same averaged
        probabilities as evaluation), and ``error_fit_info`` is added to fit_info."""
        import json
        import random
        import numpy as np
        from evaluators.error_model import (
            ERROR_MODEL_FILENAME, ERROR_MODEL_FIT_FILENAME, FINGERPRINT_METHOD, ErrorModel, error_targets,
            prediction_features, weights_fingerprint,
        )

        rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        was_training = self.model.training
        outputs, labels, n_batches = [], [], 0
        fit_loader = getattr(self, 'error_fit_loader', None)
        fit_loader = self.val_loader if fit_loader is None else fit_loader
        fit_model = getattr(self, 'error_fit_model', None)
        fit_model = self.model if fit_model is None else fit_model
        try:
            self.model.eval()
            with torch.no_grad():
                for inputs, batch_labels, *_ in fit_loader:
                    outputs.append(fit_model(inputs.to(DEVICE, non_blocking=PIN_MEM)).detach().float().cpu())
                    labels.append(batch_labels.detach().cpu())
                    n_batches += 1
        finally:
            random.setstate(rng_state[0])
            np.random.set_state(rng_state[1])
            torch.set_rng_state(rng_state[2])
            if rng_state[3] is not None:
                torch.cuda.set_rng_state_all(rng_state[3])
            self.model.train(was_training)

        out = torch.cat(outputs)
        if self.mode == 'cls':
            scores = torch.softmax(out, dim=1).numpy()
        else:
            scores = out.numpy()
        truth = torch.cat(labels).numpy()
        error_model = ErrorModel(self.mode).fit(prediction_features(scores, self.mode),
                                                error_targets(scores, truth, self.mode),
                                                num_outputs=scores.reshape(len(scores), -1).shape[1])
        error_model.fit_info.update({
            'split': 'validation', 'n_batches': n_batches,
            'val_dataset_size': len(fit_loader.dataset) if hasattr(fit_loader, 'dataset') else None,
            # The weights this model was fitted on (== what train.py saves as
            # model_weights.pth right after train()). Hash the inner model, not
            # a multi-clip wrapper, so it never depends on state_dict pass-through.
            'weights_sha256': weights_fingerprint(self.model.state_dict()),
            'weights_fingerprint_method': FINGERPRINT_METHOD,
            # Early stopping / --select_best make validation a selection set,
            # so this fit is on data the weights were chosen on (V-011, P-104).
            'validation_used_for_model_selection': bool(self.options.selects_best),
            'selection_metric': self.options.selection_metric if self.options.selects_best else None,
            'best_epoch': self.best_epoch,
        })
        error_model.fit_info.update(getattr(self, 'error_fit_info', None) or {})
        out_dir = Path(self.output_dir)
        error_model.save(out_dir / ERROR_MODEL_FILENAME)
        (out_dir / ERROR_MODEL_FIT_FILENAME).write_text(json.dumps(error_model.to_dict()['fit_info'], indent=2), encoding='utf-8')
        self.error_model = error_model
        self.logger.info(
            f"Per-prediction error model ({error_model.method_used}) fitted on {error_model.fit_info['n_samples']} "
            f"validation samples; saved to {out_dir / ERROR_MODEL_FILENAME}"
        )
        return error_model
    # --- end per-prediction error model (A4) ---

    def validate(self):
        
        """
        The validation function is devoted for the validation set only, please consider the test function for test set.
        """
        # initialize the metrics for validation
        val_accuracy=0.0
        val_loss=0.0
        total,correct=0,0
        
        
        val_pbar=tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader)
        )
        
        with torch.no_grad():
            
            for i, (input, labels,*_) in val_pbar:
                
                # set the model to evaluation mode
                self.model.eval()
                
                input, labels = input.to(DEVICE, non_blocking=PIN_MEM), labels.to(DEVICE, non_blocking=PIN_MEM)
                if self.mode == 'reg':
                    labels = labels.view(-1, 1).float()
                                
                # apply forward pass and accumulate loss

                with autocast_context(self.amp_dtype):
                    output = self.model(input)

                # Outside autocast, so in float32: the selection metric must not depend on --precision.
                loss = self.criterion(output.float(), labels)
                
                val_loss += loss.item() * labels.size(0)
                
                # calculate accuracy
                total += labels.size(0)
                
                if self.mode == 'cls':
                    correct += torch.sum(torch.argmax(output,dim=1)==labels).item()
        
        # if mode regression then no need to return accuracy or compute it
        
        if self.mode == 'cls':
            val_accuracy =  100. * correct / total
            val_loss = val_loss / total
            return val_loss, val_accuracy
        else:
            val_loss = val_loss / total
            return val_loss
            
    
    def saveModel(self, path):
        
        """
        We save the model to the path specified by the user.
        
        Args:
            path (str): The path to save the model to.
        """

        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model Saved to {path}")
