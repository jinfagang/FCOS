# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import torch
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util

from alfred.utils.log import logger as logging
from alfred.vis.image.det import visualize_det_cv2
from alfred.vis.image.get_dataset_label_map import coco_label_map_list
from alfred.dl.torch.common import device

import numpy as np


class COCODemo(object):
   
    def __init__(
        self,
        cfg,
        confidence_thresholds_for_classes,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = device
        self.model.to(self.device)
        self.min_image_size = min_image_size
        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        self.transforms = self.build_transform()
        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.cpu_device = torch.device("cpu")
        self.confidence_thresholds_for_classes = torch.tensor(confidence_thresholds_for_classes)
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        cfg = self.cfg
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)
        boxes = np.array(top_predictions.bbox.numpy(), dtype=np.int)
        scores = top_predictions.get_field('scores').numpy()
        labels = top_predictions.get_field('labels').numpy()
        scores = np.expand_dims(scores, axis=1)
        # labels not start with __background__
        labels = np.expand_dims(labels, axis=1) - 1  
        detections = np.hstack([labels, scores, boxes])
        res = visualize_det_cv2(image, detections, thresh=0.1, classes=coco_label_map_list)
        return res

    def compute_prediction(self, original_image):
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]
        prediction = predictions[0]
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        return prediction

    def select_top_predictions(self, predictions):
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels")
        thresholds = self.confidence_thresholds_for_classes[(labels - 1).long()]
        keep = torch.nonzero(scores > thresholds).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]
