from torchmetrics import MetricCollection
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
from tqdm import tqdm
import pandas as pd

import utils
from datasets.dataset_loading import CocoLoader, get_coco_split
import segmentation_models
import detection_models
import matching


def to_dict_for_map(objects, classes, scores, object_type):
    """
    Convert a list of lists of boxes (or masks) for one image into a list of dictionaries.
    The output format is needed for mAP calculation.
    Args:
        objects (list): A list of lists containing boxes or masks for one image.
        classes and scores correspond in thape to objects, and are optional
        object_type (str): The type of object to be converted. Either "boxes" or "masks".
    Returns:
        list: A list of dictionaries, where each dictionary represents a set of boxes/masks with the following keys:
            - "boxes": The list of boxes. (alternatively "masks" is returned)
            - "labels": A tensor of zeros with the same length as the list of boxes.
            - "scores": A tensor of ones with the same length as the list of boxes.

    This is needed for the MeanAveragePrecision class from torchmetrics to work right
    """
    assert object_type in ["boxes", "masks"]

    if scores is None:  # if no scores, then just ones, not important with GT
        scores = [
            torch.ones(len(instance_objects), dtype=torch.float32)
            for instance_objects in objects
        ]
    if classes is None:  # for classless metrics, all is 0
        classes = [  # generate zeros
            torch.zeros(len(instance_objects), dtype=torch.int32)
            for instance_objects in objects
        ]
    else:
        classes = [  # retype class_labels to torch int32
            torch.Tensor(instance_classes).type(torch.int32)
            for instance_classes in classes
        ]

    list_of_dicts_to_return = [
        {
            object_type: instance_objects,
            "labels": instance_classes,  # None if not given, otherwise lsit of classes
            "scores": instance_scores,  # None if not given
        }
        for instance_objects, instance_classes, instance_scores in zip(
            objects, classes, scores
        )
    ]

    return list_of_dicts_to_return


class Evaluator:
    """
    Class for evaluating object detection and segmentation models.
    Args:
        cfg: config dict
        model_seg (optional): The segmentation model to be evaluated.
        device (str, optional): The device to be used for evaluation. Defaults to "cuda".
        seg_pairwise_metrics (optional): The pairwise metrics to be calculated for segmentation.
        seg_batch_metrics (optional): The batch metrics to be calculated for segmentation.
        det_batch_metrics (optional): The batch metrics to be calculated for object detection.
        boxes_transform (optional): The transformation function to be applied to detected boxes.
        model_det (optional): The object detection model to be evaluated. If None, GT boxes are used.
        box_matching (callable, optional): The function selecting boxes for segmentation.
    """

    def __init__(
        self,
        cfg,
        model_seg=None,
        device="cuda",
        seg_pairwise_metrics=None,
        seg_batch_metrics=None,
        det_batch_metrics=None,
        boxes_transform=None,
        model_det=None,  # if None, then dummy
        box_matching: callable = None,
    ):
        self.model_det = model_det
        self.model_seg = model_seg
        self.boxes_transform = boxes_transform
        self.device = device
        self.cfg = cfg

        self.iou_boxes, self.iou_masks = [], []
        self.evaluated = False

        # batch metrics, mAP, but CA/classless
        self.det_map_classless = MeanAveragePrecision(
            iou_type="bbox",
            average="micro",
            class_metrics=False,
        )
        self.seg_map_classless = MeanAveragePrecision(
            iou_type="segm",
            average="micro",
            class_metrics=False,
        )

        # batch metrics, mAP, classful, per class, maybe extended output
        self.seg_batch_classful = MeanAveragePrecision(
            iou_type="segm", average="macro", class_metrics=True, extended_summary=False
        )
        self.det_batch_classful = MeanAveragePrecision(
            iou_type="bbox", average="macro", class_metrics=True, extended_summary=False
        )

        # calculating something at all, if det=None, then gt is used
        assert (model_seg is not None) or (model_det is not None)

    def matching(self, gt, det, det_scores, iou_type):
        """
        returns matched objects and IoUs of the matched pairs
        threshold is hardcoded to 0.5 to filter bad matches
        """
        # additional code can be wrapped here?
        # mainly need to call matching fn
        return matching.matching_fn(
            gt, det, det_scores, threshold=0.5, iou_type=iou_type
        )

    def prepare_gt(self, metadata):
        gt_boxes = [instance["boxes"] for instance in metadata]
        gt_masks = [instance["masks"].type(torch.uint8) for instance in metadata]
        gt_classes = [instance["categories"] for instance in metadata]
        return {"boxes": gt_boxes, "masks": gt_masks, "classes": gt_classes}

    def update_metrics(self, results, gt, object_type="boxes"):
        """
        Calculates segmentation/detection metrics for the given inferred and ground truth boxes.
        Args:
            results: dict with keys:
                "boxes"/"masks": list of lists of boxes or masks,
                "class_labels" list of lists of classes,
                "confidence": list of lists of confidence score for each detection
        """

        assert object_type in ["boxes", "masks"]
        detected_objects = results[object_type]
        detected_classes = results["class_labels"]
        detected_scores = results["confidence"]

        gt_objects = gt[object_type]
        gt_classes = gt["classes"]

        detected_dict = to_dict_for_map(
            objects=detected_objects,
            classes=detected_classes,
            scores=detected_scores,
            object_type=object_type,
        )
        gt_dict = to_dict_for_map(
            objects=gt_objects,
            classes=gt_classes,
            scores=None,
            object_type=object_type,
        )
        gt_classless_dict = to_dict_for_map(
            objects=gt_objects, classes=None, scores=None, object_type=object_type
        )

        # M:N metrics
        if object_type == "boxes":
            self.det_map_classless.update(preds=detected_dict, target=gt_classless_dict)
            self.det_batch_classful.update(preds=detected_dict, target=gt_dict)
        elif object_type == "masks":
            self.seg_map_classless.update(preds=detected_dict, target=gt_classless_dict)
            self.seg_batch_classful.update(preds=detected_dict, target=gt_dict)

        # 1:1 matching and IoU calculation for each image
        for batch_idx in range(len(detected_objects)):
            matched_detections, match_ious = self.matching(
                gt=gt_objects[batch_idx].cpu(),  # tensor
                det=detected_objects[batch_idx].cpu(),  # tensor
                det_scores=detected_scores[batch_idx].cpu(),  # tensor
                iou_type=object_type,
            )
            if object_type == "boxes":
                self.iou_boxes.extend(match_ious)
            elif object_type == "masks":
                self.iou_masks.extend(match_ious)

    def get_metrics(self):
        """
        Do final calculation of torchmetrics classes and wrap the results in a dict
        returns result dict, box and mask IoU arrays
        """
        assert self.evaluated, "You must first evaluate on a dataset"
        result_dict = {
            # Average IoU
            "mean det IoU": float(np.mean(np.array(self.iou_boxes))),
            "mean seg IoU": float(np.mean(np.array(self.iou_masks))),
            # mAPs , classless and per class for both seg,det
            "classless mAP - detection": utils.convert_tensors_to_save(
                self.det_map_classless.compute()
            ),
            "classless mAP - segmentation": utils.convert_tensors_to_save(
                self.seg_map_classless.compute()
            ),
            "classful mAP - detection": utils.convert_tensors_to_save(
                self.det_batch_classful.compute()
            ),
            "classful mAP - segmentation": utils.convert_tensors_to_save(
                self.seg_batch_classful.compute()
            ),
        }
        return (
            result_dict,
            np.array(self.iou_masks),
            np.array(self.iou_boxes),
            np.array([]),  # TODO: for now, index array? pd frame?
        )

    def evaluate(self, data_loader, savepath=None, max_batch=None):
        """
        Evaluate the performance of the model on a given data loader.
        Args:
            data_loader (torch.utils.data.DataLoader): The data loader containing the evaluation dataset.
            savepath (str, optional): The path to save the evaluation results. Defaults to None.
            max_batch (int, optional): The maximum number of batches to evaluate. Defaults to None.
                if None, then whole dataset is evaluated.
        Returns:
            None
            Metrics are stored in the torchmetrics classes, and can be accessed with get_metrics()
        """

        for batch_idx, batch in tqdm(enumerate(data_loader)):
            if (max_batch is not None) and (batch_idx >= max_batch):
                break
            images = list(batch[0])
            metadata = list(batch[1])

            # list of list of mask/boxes and classes as GT
            gt = self.prepare_gt(metadata)

            # detection module. attention points and labels possible with detection classes
            detection_results = self.model_det.detect_batch(
                images, metadata
            )  # need metadata for GT

            # detection metrics
            self.update_metrics(results=detection_results, gt=gt, object_type="boxes")

            # can do box transforms here
            if self.boxes_transform is not None:
                detected_boxes = boxes_transform(detected_boxes)

            # segment, calculate metrics
            if self.model_seg is not None:
                segmentation_results = self.model_seg.infer_batch(
                    images, detection_results
                )

                # if segmentation, then calculate metrics
                segmentation_results["class_labels"] = detection_results["class_labels"]

                self.update_metrics(
                    results=segmentation_results, gt=gt, object_type="masks"
                )

        # I can return the metrics after this, otherwise error
        self.evaluated = True


def test_evaluator():
    print("\nTesting Evaluator")
    device = "cpu"

    detector = detection_models.GTboxDetector()
    segmentation_model = segmentation_models.BaseSegmentWrapper(
        device=device, model="whatever, dummy class here"
    )
    boxes_transform = None

    evaluator = Evaluator(
        device=device,
        cfg=None,
        model_det=detector,
        model_seg=segmentation_model,
        boxes_transform=boxes_transform,
    )
    print("Evaluator test passed!")


if __name__ == "__main__":
    test_evaluator()
