from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import sys
import logging
from helpers import PerformanceLogger
from options import DEVICE, PIN_MEM, TRAINVAL_OUTPUT_DIR
import time
from utils import save_process_times
import functools
print = functools.partial(print, flush=True)


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float,
        mode: str,
        num_epochs: int,
        output_dir: Path = TRAINVAL_OUTPUT_DIR
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
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if not params:
            raise ValueError("No trainable parameters found in the model!")
        self.optimizer = optim.Adam(params, lr=self.learning_rate)
        
        self.performance_logger = PerformanceLogger(output_dir) if output_dir else None

        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()

        # --- trainable / total parameter report (A1 param budget) ---
        import json
        from trainers.param_budget import summarize_parameters, format_summary
        self.parameter_summary = summarize_parameters(self.model)
        self.logger.info(format_summary(self.parameter_summary))
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(output_dir) / 'parameter_summary.json', 'w', encoding='utf-8') as stream:
                json.dump(self.parameter_summary, stream, indent=2)
        # --- end parameter report ---

    def train(self):
                    
        total_time = 0
        epoch_times = []
        for epoch in range(self.num_epochs):

            self.model.train()
            epoch_start = time.time()
            
            
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
                    

                # null the gradient
                self.optimizer.zero_grad()

                # run the model
                outputs = self.model(inputs)

                # compute loss
                loss = self.criterion(outputs, labels)

                # backprop and apply optimizer
                loss.backward()
                self.optimizer.step()

                # accumulate loss
                running_loss += loss.item() * labels.size(0)
                samples_seen += labels.size(0)
                
                if self.mode == "cls":
                
                    # calculate accuracy
                    correct_predictions += torch.sum(torch.argmax(outputs,dim=1)==labels).item()
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
                val_loss, val_accuracy = self.validate()
                self.logger.info(f"Validation loss: {val_loss:.3f} , Valiadation Accuracy: {val_accuracy:.3f} % ")
            
            else:
                """
                Only loss is needed to be logged in regression mode.
                """
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {epoch_loss}"
                )    
    
                val_loss = self.validate()    
            
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
        # record the total training time at the end
        save_process_times(epoch_times, total_time, self.output_dir,"training")

        # --- per-prediction error model (A4) ---
        # One extra pass over the VALIDATION loader (never test data) with the
        # final weights -- the ones train.py saves as model_weights.pth -- fits
        # error_model.json next to them. Opt out with --no_error_model (or
        # trainer.fit_error_model = False). RNG state and train/eval mode are
        # restored, so training results are unchanged.
        enabled = getattr(self, 'fit_error_model', None)
        if enabled is None:
            from evaluators.error_model import fit_by_default
            enabled = fit_by_default()
        if enabled and self.output_dir:
            try:
                self._fit_error_model()
            except Exception as exc:  # never let the auxiliary estimate break a finished training run
                self.logger.warning(f'Per-prediction error model was not fitted: {exc}')
        # --- end per-prediction error model (A4) ---

    # --- per-prediction error model (A4) ---
    def _fit_error_model(self):
        """Fit the per-prediction error model on the validation split and
        save ``error_model.json`` + ``error_model_fit.json`` in output_dir.
        Returns the fitted ``evaluators.error_model.ErrorModel``."""
        import json
        import random
        import numpy as np
        from evaluators.error_model import (
            ERROR_MODEL_FILENAME, ERROR_MODEL_FIT_FILENAME, ErrorModel, error_targets, prediction_features,
        )

        rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                     torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        was_training = self.model.training
        outputs, labels, n_batches = [], [], 0
        try:
            self.model.eval()
            with torch.no_grad():
                for inputs, batch_labels, *_ in self.val_loader:
                    outputs.append(self.model(inputs.to(DEVICE, non_blocking=PIN_MEM)).detach().float().cpu())
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
            'val_dataset_size': len(self.val_loader.dataset) if hasattr(self.val_loader, 'dataset') else None,
        })
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
                
                output = self.model(input)
                
                loss = self.criterion(output, labels)
                
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
