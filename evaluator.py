from torchmetrics import MetricCollection
import torch
import torchvision
from torchmetrics.classification import JaccardIndex
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
from tqdm import tqdm
import math

import utils  # contains sam_utils, visual_utils, and other utility functions
from datasets.dataset_loading import CocoLoader, get_coco_split
import segmentation_models
import detection_models


# helper functions for evaluator list of lists of boxs/masks -> list of dicts with boxes/masks
def to_dict_map(list_of_list_of_boxes):
    """
    Convert a list of lists of boxes for one image into a list of dictionaries.
    Args:
        list_of_list_of_boxes (list): A list of lists containing boxes for one image.
    Returns:
        list: A list of dictionaries, where each dictionary represents a box with the following keys:
            - "boxes": The list of boxes.
            - "labels": A tensor of zeros with the same length as the list of boxes.
            - "scores": A tensor of ones with the same length as the list of boxes.
    """
    key_type = (
        "boxes" if len(list_of_list_of_boxes[0][0]) == 4 else "masks"
    )  # last dim is 4 -> boxes, else masks

    return [
        {
            key_type: list_of_boxes,
            "labels": torch.zeros(len(list_of_boxes), dtype=torch.int32),
            "scores": torch.ones(len(list_of_boxes), dtype=torch.float32),
        }
        for list_of_boxes in list_of_list_of_boxes
    ]


def to_dict_map_classful(list_of_list_of_boxes, list_of_list_of_classes):
    """
    Convert a list of lists of boxes for one image into a list of dictionaries.
    Args:
        list_of_list_of_boxes (list): A list of lists containing boxes for one image.
    Returns:
        list: A list of dictionaries, where each dictionary represents a box with the following keys:
            - "boxes": The list of boxes.
            - "labels": A tensor of zeros with the same length as the list of boxes.
            - "scores": A tensor of ones with the same length as the list of boxes.
    """
    key_type = (
        "boxes" if len(list_of_list_of_boxes[0][0]) == 4 else "masks"
    )  # last dim is 4 -> boxes, else masks

    return [
        {
            key_type: list_of_boxes,
            "labels": torch.Tensor(list_of_classes).type(torch.int32),
            "scores": torch.ones(len(list_of_boxes), dtype=torch.float32),
        }
        for list_of_boxes, list_of_classes in zip(
            list_of_list_of_boxes, list_of_list_of_classes
        )
    ]


class Evaluator:
    """
    Class for evaluating object detection and segmentation models.
    Args:
        model_seg (optional): The segmentation model to be evaluated.
        device (str, optional): The device to be used for evaluation. Defaults to "cuda".
        seg_pairwise_metrics (optional): The pairwise metrics to be calculated for segmentation.
        seg_batch_metrics (optional): The batch metrics to be calculated for segmentation.
        det_batch_metrics (optional): The batch metrics to be calculated for object detection.
        boxes_transform (optional): The transformation function to be applied to detected boxes.
        model_det (optional): The object detection model to be evaluated. If None, GT boxes are used.
        box_matching (callable, optional): The function selecting boxes for segmentation.
    Attributes:
        model_det: The object detection model.
        model_seg: The segmentation model.
        det_batch_metrics: The batch metrics for object detection.
        seg_batch_metrics: The batch metrics for segmentation.
        seg_pairwise_metrics: The pairwise metrics for segmentation.
        boxes_transform: The transformation function for detected boxes.
        box_matching: The function to match ground truth boxes with detected boxes.
        device: The device used for evaluation.
    Methods:
        evaluate(data_loader, savepath=None, max_batch=None):
            Evaluates the models using the given data loader.
    Usage:
        evaluator = Evaluator(model_seg, device="cuda", seg_pairwise_metrics, seg_batch_metrics, det_batch_metrics,
                            boxes_transform, model_det, box_matching)
        evaluator.evaluate(data_loader, savepath=None, max_batch=None)
    """

    def __init__(
        self,
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

        self.IoU_boxes = []
        self.IoU_masks = []
        self.evaluated = False

        # pairwise metrics, wIoU
        self.seg_IoU_metric = JaccardIndex("binary").to(device)
        self.det_IoU_metric = JaccardIndex("binary").to(device)

        # batch metrics, mAP, but CA/classless
        self.det_map_classless = MeanAveragePrecision(
            iou_type="bbox",
            average="macro",
            class_metrics=False,
        )
        self.seg_map_classless = MeanAveragePrecision(
            iou_type="segm",
            average="micro",
            class_metrics=False,
        )

        # batch metrics, mAP, classful, per class, extended output
        self.seg_batch_classful = MeanAveragePrecision(
            iou_type="segm", average="micro", class_metrics=True, extended_summary=True
        )
        self.det_batch_classful = MeanAveragePrecision(
            iou_type="bbox", average="macro", class_metrics=True, extended_summary=True
        )

        # calculating something at all
        assert (model_seg is not None) or (model_det is not None)

    def matching(self, gt, inferred):
        # TODO implement actual matching.
        # how do they do it in torchmetrics mAP?
        length = len(gt)  # for now just take the same amount of boxes and call it a day
        return inferred[:length]

    def boxes_to_masks(self, boxes, mask_shape):
        masks = []
        for box in boxes:
            mask = torch.zeros(mask_shape, dtype=torch.uint8)
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)
        return masks

    def prepare_gt(self, metadata):
        gt_boxes = [instance["boxes"] for instance in metadata]
        gt_masks = [instance["masks"].type(torch.uint8) for instance in metadata]
        gt_classes = [instance["categories"] for instance in metadata]
        return gt_boxes, gt_masks, gt_classes

    def calculate_metrics_detection(
        self, detected_boxes, detected_classes, gt_boxes, gt_classes, images_shapes
    ):
        """
        Calculates segmentation metrics for the given inferred and ground truth boxes.
        Needs image shapes when converting the boxes to masks for jaccard index calculation.
        Args:
            detected_classes (list): List of inferred classes, as they come from boxes
            gt_class (list): List of ground truth classes.
            detected_boxes (list): List of lists of inferred boxes.
            gt_boxes (list): List of lists of ground truth boxes.
        Returns:
            None
        """
        # M:N metrics
        self.det_map_classless.update(
            to_dict_map(detected_boxes),
            to_dict_map(gt_boxes),
        )

        self.det_batch_classful.update(
            to_dict_map_classful(detected_boxes, detected_classes),
            to_dict_map_classful(gt_boxes, gt_classes),
        )

        # 1:1 metrics
        selected_boxes = self.matching(gt=gt_boxes, inferred=detected_boxes)
        for gt_box_list, box_list, shape in zip(
            gt_boxes, selected_boxes, images_shapes
        ):
            # convert boxes to masks using original shape
            box_list = self.boxes_to_masks(box_list, shape)
            gt_box_list = self.boxes_to_masks(gt_box_list, shape)

            # pair boxes together
            for gt, inferred in zip(
                gt_box_list,
                box_list,
            ):
                IoU = self.det_IoU_metric.forward(
                    inferred.to(self.device), gt.to(self.device)
                )
                self.IoU_boxes.append(IoU.cpu())

    def calculate_metrics_segmentation(
        self,
        inferred_masks,
        inferred_classes,
        gt_masks,
        gt_classes,
    ):
        """
        Calculates segmentation metrics for the given inferred and ground truth masks.
        Args:
            inferred_class (list): List of inferred classes, as they come from boxes
            gt_class (list): List of ground truth classes.
            inferred_masks (list): List of lists of inferred masks.
            gt_masks (list): List of lists of ground truth masks.
        Returns:
            None
        """

        # M:N metrics
        self.seg_map_classless.update(
            to_dict_map(inferred_masks),
            to_dict_map(gt_masks),
        )

        self.seg_batch_classful.update(
            to_dict_map_classful(inferred_masks, inferred_classes),
            to_dict_map_classful(gt_masks, gt_classes),
        )

        # 1:1 metrics
        selected_masks = self.matching(gt=gt_masks, inferred=inferred_masks)
        for gt_masks_list, masks_list in zip(gt_masks, selected_masks):
            # pair masks together
            for gt, inferred in zip(
                gt_masks_list.to(self.device),
                masks_list.to(self.device),
            ):
                IoU = self.seg_IoU_metric.forward(inferred, gt)
                self.IoU_masks.append(IoU.cpu())

    def get_metrics(self):
        assert self.evaluated

        # TODO: return all in dicts, not just the IoUs
        # compose dicts with metrics
        # calculate mean IoUs
        result_dict = {
            # weighted avg IoUs and equal weight (mean) IoUs
            "weighted det IoU": self.det_IoU_metric.compute(),
            "weighted seg IoU": self.seg_IoU_metric.compute(),
            "mean det IoU": np.mean(np.array(self.IoU_boxes)),
            "mean seg IoU": np.mean(np.array(self.IoU_masks)),
            # whole arrays
            "IoU array boxes": self.IoU_boxes,
            "IoU array masks": self.IoU_masks,
            # mAPs , classless and per class for both seg,det
            "classless mAP - detection": self.det_map_classless.compute(),
            "classless mAP - segmentation": self.seg_map_classless.compute(),
            "classful mAP - detection": self.det_batch_classful.compute(),
            "classful mAP - segmentation": self.seg_batch_classful.compute(),
        }
        return result_dict

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
            Metrics are stored in the torchmetrics classes
        Raises:
            None
        """

        # if classes are needed for detection (DINO etc.)
        classes = data_loader.dataset.get_classes()

        for i, batch in tqdm(enumerate(data_loader)):
            if (max_batch is not None) and (i > max_batch):
                break
            images = list(batch[0])
            metadata = list(batch[1])

            # list of list of mask/boxes as GT
            gt_boxes, gt_masks, gt_classes = self.prepare_gt(metadata)

            # detection
            detected_boxes, attention_points, point_labels, detection_classes = (
                self.model_det.detect_batch(images, metadata, classes)
            )

            # detection metrics
            self.calculate_metrics_detection(
                detected_boxes=detected_boxes,
                detected_classes=detection_classes,
                gt_classes=gt_classes,
                gt_boxes=gt_boxes,
                images_shapes=[gt_mask.shape for gt_mask in gt_masks],
            )

            # can do box transforms here
            if self.boxes_transform is not None:
                detected_boxes = boxes_transform(detected_boxes)

            # segment, calculate metrics
            if self.model_seg is not None:
                inferred_masks = self.model_seg.infer_batch(
                    images, detected_boxes, attention_points, point_labels
                )

                # if segmentation, then calculate metrics
                self.calculate_metrics_segmentation(
                    inferred_masks=inferred_masks,
                    inferred_classes=detection_classes,
                    gt_classes=gt_classes,
                    gt_masks=gt_masks,
                )

        # I can return the metrics after this
        self.evaluated = True


def test_evaluator():
    print("Not implemented yet")


if __name__ == "__main__":
    test_evaluator()
