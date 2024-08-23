import torch
import utils
import numpy as np


class BaseDetectorWrapper:

    def detect_batch(self, images, metadata, classes=None):
        raise NotImplementedError


class DummyDetectorBase(BaseDetectorWrapper):

    def extract_GT_boxes(self, metadata_list):
        results = [torch.Tensor(instance["boxes"]) for instance in metadata_list]
        return results  # returns list of lists of boxes from metadata


class GTboxDetector(DummyDetectorBase):
    """
    Dummy detector class, which just takes GT bboxes with no other prompts
    """

    def detect_batch(self, images, metadata, classes=None):
        results = self.extract_GT_boxes(metadata)
        return (
            results,  # the GT boxes extracted from metadata
            None,  # no attention points
            None,  # no point labels
            [[0 for box in boxes] for boxes in results],  # returns all classes as 0,
        )
        # gives no point labels or attention points, simply GT boxes


class GTboxDetectorMiddle(DummyDetectorBase):
    """
    Dummy detector class,
    takes the middle point as a positive prompt to sam
    """

    def detect_batch(self, images, metadata=None, classes=None):
        results = self.extract_GT_boxes(metadata)  # list of lists of boxes
        classes = [
            [0 for box in boxes] for boxes in results
        ]  # returns all classes as 0

        points = []
        prompts = []
        for boxes in results:  # not batching, sequentially load and middles
            [points.append([utils.get_middle_point(box) for box in boxes])]
            [prompts.append([1 for box in boxes])]
        return results, points, prompts, classes
