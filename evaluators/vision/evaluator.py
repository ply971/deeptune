import json
import logging
import sys
import torch
import torch.nn as nn
from utils import save_process_times
from sklearn.metrics import classification_report, roc_auc_score
from tqdm.auto import tqdm
import time

from options import DEVICE


class TestTrainer:
    """
    Performs Testing on the input image dataset.
    
    Attributes:
    
            model (PyTorch Model): The model we are loading from the src file, whether it is for transfer learning with PEFT Or without.
            test_loader (torch.utils.data.DataLoader): The DataLoader for the test set.
            batch_size (int): The batch size for the test set.
            criterion (torch.nn.Module): Loss function, Cross Entropy as we do classification.
            performance_logger (PerformanceLogger): Logger instance for tracking testing.
            logger (logging.Logger): Logger instance for tracking test progress.
    """
    
    def __init__(
        self,
        model,
        batch_size,
        test_loader,
        mode,
        output_dir,
        device=DEVICE
    ):
        self.model = model
        self.batch_size = batch_size
        self.test_loader = test_loader
        self.mode = mode

        self.output_dir = output_dir

        self.device = device
        
        # logging info
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s")
        self.logger = logging.getLogger()
        
        if self.mode == 'cls':    
            self.criterion = nn.CrossEntropyLoss()
        else: # then regression if not classification
            self.criterion = nn.MSELoss()
        
        
    def test(self, best_model_weights_path=None, error_model_path=None):
        """Score the test loader, write ``full_metrics.json`` and a per-sample
        ``predictions.csv``. When the run has an ``error_model.json`` (next to
        ``best_model_weights_path``, or ``error_model_path``), predictions
        get an ``expected_error`` column and ``error_calibration.json/.csv``
        report stated vs observed error per decile on this held-out set."""
        start_time = time.time()
        if best_model_weights_path is not None:
            self.model.load_state_dict(torch.load(best_model_weights_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        total_loss, total = 0.0, 0
        all_labels, all_predictions, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels, *_ in tqdm(self.test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.mode == 'reg':
                    labels = labels.view(-1, 1).float()
                outputs = self.model(inputs)
                total_loss += self.criterion(outputs, labels).item() * labels.size(0)
                total += labels.size(0)
                all_labels.append(labels.detach().cpu())
                if self.mode == 'cls':
                    probabilities = torch.softmax(outputs, dim=1)
                    all_probs.append(probabilities.cpu())
                    all_predictions.append(probabilities.argmax(dim=1).cpu())
                else:
                    all_predictions.append(outputs.detach().cpu())
        if not total:
            raise ValueError('The evaluation dataset is empty.')
        metrics = {'loss': total_loss / total}
        labels = torch.cat(all_labels).numpy()
        predictions = torch.cat(all_predictions).numpy()
        if self.mode == 'cls':
            probabilities = torch.cat(all_probs).numpy()
            metrics.update(classification_report(labels, predictions, output_dict=True, zero_division=0))
            try:
                scores = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities
                metrics['auroc'] = float(roc_auc_score(labels, scores, multi_class='ovr'))
            except ValueError:
                metrics['auroc'] = None
        else:
            from sklearn.metrics import mean_absolute_error, r2_score
            metrics['mae'] = float(mean_absolute_error(labels, predictions))
            metrics['rmse'] = float(metrics['loss'] ** 0.5)
            metrics['r2'] = float(r2_score(labels, predictions)) if total >= 2 else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # --- per-sample predictions + per-prediction error estimate (A4) ---
        scores = probabilities if self.mode == 'cls' else predictions
        per_sample = _per_sample_predictions(labels, predictions, scores, self.mode)
        estimate = _attach_error_estimate(per_sample, scores, self.mode, self.output_dir,
                                          best_model_weights_path, error_model_path, self.logger)
        if estimate is not None:
            metrics['error_estimate'] = estimate
        per_sample.to_csv(self.output_dir / PREDICTIONS_FILENAME, index=False)
        # --- end (A4) ---
        with open(self.output_dir / 'full_metrics.json', 'w', encoding='utf-8') as stream:
            json.dump(metrics, stream, indent=2)
        save_process_times(epoch_times=1, total_duration=time.time() - start_time,
                           outdir=self.output_dir, process='evaluation')
        return metrics


# --- per-sample predictions + per-prediction error estimate (A4) ---
PREDICTIONS_FILENAME = 'predictions.csv'
MAX_PROB_COLUMNS = 50  # wider class sets skip the per-class prob_<k> columns


def _per_sample_predictions(labels, predictions, scores, mode):
    """One row per test sample, in loader order."""
    import numpy as np
    import pandas as pd
    n = len(labels)
    frame = pd.DataFrame({'sample_index': np.arange(n)})
    if mode == 'cls':
        truth = labels.reshape(-1).astype(np.int64)
        pred = predictions.reshape(-1).astype(np.int64)
        frame['true_label'] = truth
        frame['predicted_class'] = pred
        frame['confidence'] = scores.max(axis=1)
        frame['correct'] = truth == pred
        if scores.shape[1] <= MAX_PROB_COLUMNS:
            for k in range(scores.shape[1]):
                frame[f'prob_{k}'] = scores[:, k]
    else:
        values = predictions.reshape(n, -1)
        truth = labels.reshape(n, -1).astype(np.float64)
        if values.shape[1] == 1:
            frame['true_value'] = truth[:, 0]
            frame['predicted_value'] = values[:, 0]
        frame['abs_error'] = np.abs(values - truth).mean(axis=1)
    return frame


def _attach_error_estimate(per_sample, scores, mode, output_dir, checkpoint_path, error_model_path, logger):
    """Adds ``expected_error`` and writes error_calibration.json/.csv when the
    run has an error model; returns the summary for full_metrics.json (or
    None for old runs without one)."""
    from evaluators.error_model import (
        CALIBRATION_STEM, calibration_table, load_error_model_near, metrics_summary,
        observed_errors, prediction_features, save_report,
    )
    num_outputs = scores.reshape(len(scores), -1).shape[1]
    model = load_error_model_near(checkpoint_path, mode, num_outputs=num_outputs,
                                  path=error_model_path, warn=logger.warning)
    if model is None:
        return None
    per_sample['expected_error'] = model.predict(prediction_features(scores, mode))
    report = calibration_table(per_sample['expected_error'].to_numpy(), observed_errors(per_sample, mode), mode=mode)
    save_report(report, output_dir, CALIBRATION_STEM,
                extra={'split': 'test', 'error_model': {k: model.fit_info.get(k) for k in
                                                        ('method', 'n_samples', 'base_error', 'split', 'score_feature')}})
    return metrics_summary(report, model)
