from datasets.dataset_loading import CocoLoader
import utils


import torch
import numpy as np

# sam
from segment_anything import SamPredictor, sam_model_registry


"""
Main pipeline logic. 
load datasets
load detection models
load segmentation models

test detection on dataset, comapre with GT bboxes
test segmentation on dataset, compare with GT masks

Goal: 
    - metrics for detection
    - metrics for segmentation
    - output tables, output visualizations, saving BBoxes, masks, etc.
    - eventually another script to load the results without rerunning
    - reproducible code here, examples of usage
    - validation and training sets (just inference here)
"""
if __name__ == "__main__":
    assert torch.cuda.is_available()

    coco = CocoLoader()
    transforms = None  # No augmentations for now
    data_train, api = coco.load_train(transformations=transforms)

    # load segmentation model(s)
    predictor, sam = utils.prepare_sam("cuda", model="s")

    successful_boxes = 0
    thresholds = []  # for recall?
