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
import matching


# TODO: move to utils??
def filter_images(images, metadata):
    # Calculate where_zero array, 1 if no boxes, 0 if boxes
    filtered_images = []
    filtered_metadata = []

    for j in range(len(images)):
        if len(metadata[j]["boxes"]) == 0:  # no boxes?
            continue
        filtered_images.append(images[j])
        filtered_metadata.append(metadata[j])
    return filtered_images, filtered_metadata


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
        scores = [None for _ in objects]
    if classes is None:  # for classless metrics
        classes = [None for _ in objects]

    return [
        {
            object_type: instance_boxes,
            "labels": torch.Tensor(instance_classes).type(torch.int32),
            "scores": instance_scores,  # None if not given
        }
        for instance_objects, instance_classes, instance_scores in zip(
            objects, classes, scores
        )
    ]


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

        self.iou_boxes, self.iou_masks = [], []

        self.evaluated = False
        self.cfg = cfg

        # pairwise metrics, wIoU
        self.seg_iou_metric = JaccardIndex("binary").to(device)
        self.det_iou_metric = JaccardIndex("binary").to(device)

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
        # TODO implement actual matching.
        # call matching.matching_fn(gt, det, det_scores, threshold=0.5, iou_type=iou_type)

        length = len(gt)  # for now just take the same amount of boxes and call it a day
        return inferred[:length]

    def prepare_gt(self, metadata):
        gt_boxes = [instance["boxes"] for instance in metadata]
        gt_masks = [instance["masks"].type(torch.uint8) for instance in metadata]
        gt_classes = [instance["categories"] for instance in metadata]
        return {"boxes": gt_boxes, "masks": gt_masks, "classes": gt_classes}

    def calculate_metrics_detection(self, detection_results, gt):
        """
        Calculates segmentation metrics for the given inferred and ground truth boxes.
        Needs image shapes when converting the boxes to masks for jaccard index calculation.
        Args:
            detection_results: dict with keys:
                "boxes": list of lists of boxes,
                "class_labels" list of lists of classes,
                "confidence": list of lists of confidence score for each detection
                "attention_points": list of lists of attention points for each detection
                "point_labels": for each attention point,
                        the label means if it is positive or negative prompt
        Returns:
            None
        """
        detected_boxes = detection_results["boxes"]
        detected_classes = detection_results["class_labels"]
        detected_scores = detection_results["confidence"]

        gt_boxes = gt["boxes"]
        gt_classes = gt["classes"]

        detected_dict = to_dict_for_map(
            objects=detected_boxes,
            classes=detected_classes,
            scores=detected_scores,
            object_type="boxes",
        )
        gt_dict = to_dict_for_map(
            objects=gt_boxes,
            classes=gt_classes,
            scores=None,
            object_type="boxes",
        )
        gt_classless_dict = to_dict_for_map(
            objects=gt_boxes, classes=None, scores=None, object_type="boxes"
        )

        # M:N metrics
        self.det_map_classless.update(detected_dict, gt_classless_dict)
        self.det_batch_classful.update(detected_dict, gt_dict)

        # 1:1 metrics
        matched_detections, match_ious = self.matching(
            gt=gt_boxes,
            det=detected_boxes,
            det_scores=detected_scores,
            iou_type="boxes",
        )
        self.iou_boxes.extend(match_ious)

        # # TODO: dont need this??, just extend self.iou_boxes and save to pandas?
        # for gt_box_list, box_list, shape in zip(
        #     gt_boxes, selected_boxes, images_shapes
        # ):
        #     if len(gt_box_list) == 0:
        #         continue

        #     # convert boxes to masks using original shape
        #     box_list_as_mask = utils.boxes_to_masks(box_list, shape)
        #     gt_box_list_as_mask = utils.boxes_to_masks(gt_box_list, shape)

        #     # pair boxes together
        #     for gt, inferred in zip(
        #         box_list_as_mask,
        #         gt_box_list_as_mask,
        #     ):
        #         iou = self.det_iou_metric.forward(
        #             inferred.to(self.device), gt.to(self.device)
        #         )
        #         self.iou_boxes.append(iou.cpu())

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
        # this is a bit inefficien, but works for now (same as in boxes)
        zeros_as_gt = [[0 for _ in gt_box_list] for gt_box_list in gt_classes]
        #       to properly register classless metrics, we invent a new dummy GT classes list
        #       this means everything is in one class
        # there probably is a way to refactor hese 2 functions into one intelligent function for boxes and masks,
        #           no time rn!, TODO later

        dict_detected = to_dict_for_map(inferred_masks, inferred_classes)
        dict_gt = to_dict_for_map(gt_masks, gt_classes)
        dict_gt_classless = to_dict_for_map(gt_masks, zeros_as_gt)

        # M:N metrics
        self.seg_map_classless.update(dict_detected, dict_gt_classless)
        self.seg_batch_classful.update(dict_detected, dict_gt)

        # 1:1 metrics
        selected_masks = self.matching(gt=gt_masks, inferred=inferred_masks)
        for gt_masks_list, masks_list in zip(gt_masks, selected_masks):
            # pair masks together
            for gt, inferred in zip(
                gt_masks_list.to(self.device),
                masks_list.to(self.device),
            ):
                iou = self.seg_iou_metric.forward(inferred, gt)
                self.iou_masks.append(iou.cpu())

    def get_metrics(self):
        """
        Do final calculation of torchmetrics classes and wrap the results in a dict
        returns result dict, box and mask IoU arrays
        """
        assert self.evaluated
        result_dict = {
            # weighted avg IoUs and equal weight (mean) IoUs
            "weighted det IoU": float(self.det_iou_metric.compute().cpu()),
            "weighted seg IoU": float(self.seg_iou_metric.compute().cpu()),
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

        for i, batch in tqdm(enumerate(data_loader)):
            if (max_batch is not None) and (i > max_batch):
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
            self.calculate_metrics_detection(detection_results=detection_results, gt=gt)

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
                    inferred_classes=detection_results["classes"],
                    gt_classes=gt["classes"],
                    gt_masks=gt["masks"],
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
        model_det=detector,
        model_seg=segmentation_model,
        boxes_transform=boxes_transform,
    )
    print("Evaluator test passed!")


if __name__ == "__main__":
    test_evaluator()
