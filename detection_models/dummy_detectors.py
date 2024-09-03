import torch
import utils
import numpy as np


class BaseDetectorWrapper:
    def __init__(self, cfg=None, device=None, all_classes=None):
        pass

    def detect_batch(self, images, metadata):
        raise NotImplementedError


class GTDetectorBase(BaseDetectorWrapper):

    def extract_GT_boxes(self, metadata_list):
        results = [torch.Tensor(instance["boxes"]) for instance in metadata_list]
        return results  # returns list of lists of boxes from metadata
        # [] when empty


class GTboxDetector(GTDetectorBase):
    """
    Dummy detector class, which just takes GT bboxes with no other prompts
    """

    def detect_batch(self, images, metadata):
        # extract gt boxes
        results = self.extract_GT_boxes(metadata)

        # fabricate scores and classes=0, no attention points (and labels for them )
        dict_res = {
            "boxes": results,
            "class_labels": [
                torch.zeros(len(boxes), dtype=torch.int16) for boxes in results
            ],
            "confidence": [
                torch.ones(len(boxes), dtype=torch.float32) for boxes in results
            ],
            "attention_points": None,
            "point_labels": None,
        }
        return dict_res


class GTboxDetectorMiddle(GTDetectorBase):
    """
    Dummy detector class,
    takes the middle point as a positive prompt to sam
    """

    def detect_batch(self, images, metadata=None):
        # extract gt boxes
        results = self.extract_GT_boxes(metadata)  # list of lists of boxes

        # extract middle points
        points, prompts = [], []
        for boxes in results:  # not batching, sequentially load and middles
            [points.append([utils.get_middle_point(box) for box in boxes])]
            [prompts.append([1 for box in boxes])]

        # fabricate scores and classes=0
        results_dict = {
            "boxes": results,
            "attention_points": points,  # middle here
            "point_labels": prompts,  # positive prompt label
            "class_labels": [
                torch.zeros(len(boxes), dtype=torch.int16) for boxes in results
            ],
            "confidence": [
                torch.ones(len(boxes), dtype=torch.float32) for boxes in results
            ],
        }

        return results_dict
