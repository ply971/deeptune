"""Local, fine-tuned torchvision detector. Importing this module is torch-free."""
from pathlib import Path

import numpy as np

from spatial.detect import Detector, finalize_detections, pick_device

ARCHITECTURE = 'fasterrcnn_mobilenet_v3_large_320_fpn'


def make_model(pretrained=False):
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights='DEFAULT' if pretrained else None, weights_backbone=None)
    old = model.roi_heads.box_predictor
    head = FastRCNNPredictor(old.cls_score.in_features, 2)
    if pretrained:
        # COCO category 17 is cat. Preserve the baseline before fine-tuning.
        with torch.no_grad():
            for new, previous in enumerate((0, 17)):
                head.cls_score.weight[new].copy_(old.cls_score.weight[previous])
                head.cls_score.bias[new].copy_(old.cls_score.bias[previous])
                head.bbox_pred.weight[new*4:new*4+4].copy_(old.bbox_pred.weight[previous*4:previous*4+4])
                head.bbox_pred.bias[new*4:new*4+4].copy_(old.bbox_pred.bias[previous*4:previous*4+4])
    model.roi_heads.box_predictor = head
    return model


class PetDetector(Detector):
    name = 'pet_fasterrcnn'

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        import torch
        path = Path(model_id)
        if not path.is_file():
            raise ValueError(f'Choose a trained cat checkpoint (.pt): {path}')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('architecture') != ARCHITECTURE or checkpoint.get('labels') != ['background', 'cat']:
            raise ValueError('Unsupported cat detector checkpoint architecture or labels')
        model = make_model()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        return cls(model, model_id=str(path.resolve()), metadata=checkpoint.get('provenance', {}), **kwargs)

    def __init__(self, model, model_id='', metadata=None, device='auto', box_threshold=0.15,
                 batch_size=1, nms_iou=0.5):
        super().__init__(batch_size, nms_iou)
        self.device = pick_device(device)
        self.model = model.to(self.device).eval()
        self.model_id = model_id
        self.metadata = metadata or {}
        self.box_threshold = float(box_threshold)

    def _detect_batch(self, frames, prompts, frame_indices):
        import torch
        if any(p.lower() != 'cat' for p in prompts):
            raise ValueError('This trained detector supports only the prompt cat.')
        tensors = [torch.from_numpy(np.array(f, copy=True)).permute(2, 0, 1).float().div(255).to(self.device)
                   for f in frames]
        with torch.inference_mode():
            predictions = self.model(tensors)
        result = []
        for frame, pred in zip(frames, predictions):
            keep = (pred['labels'] == 1) & (pred['scores'] >= self.box_threshold)
            boxes = pred['boxes'][keep].cpu().numpy()
            scores = pred['scores'][keep].cpu().numpy()
            result.append(finalize_detections(boxes, scores, [prompts[0]] * len(scores),
                                             frame.shape[1], frame.shape[0], self.nms_iou))
        return result

    def describe(self):
        return {'name': self.name, 'model_id': self.model_id, 'labels': ['cat'],
                'architecture': ARCHITECTURE, 'device': self.device, 'provenance': self.metadata}
