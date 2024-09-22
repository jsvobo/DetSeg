from torchmetrics import MetricCollection
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
from tqdm import tqdm
import pandas as pd

import utils
from datasets import CocoLoader, get_coco_split, ImagenetLoader, get_imagenet_split
import segmentation_models
import detection_models
import matching
import json


def _compute_metrics_for_frame(frame, number_detections):
    number_GT = len(frame)
    matched = frame[frame["iou"] != 0]
    not_matched = frame[frame["iou"] == 0]

    TP = len(matched)  # matched GT
    FN = len(not_matched)  # missed GT
    FP = number_detections - TP  # how many detection did not match with anything?

    recall = TP / number_GT if number_GT != 0 else 0  # TP / all GTs
    precision = (
        TP / number_detections if number_detections != 0 else 0
    )  # TP / all detections
    f1 = (2 * TP) / (2 * TP + FP + FN)  # alternative to calculate F1

    metrics = {
        "TP": int(TP),
        "FN": int(FN),
        "FP": int(FP),
        "Recall": float(recall),
        "Precision": float(precision),
        "F1": float(f1),
    }
    return metrics


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
        saver,
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
        self.image_counter = 0
        self.saver = saver

        # dicts for saving detection and segmentation metadata
        self.boxes_dict = {
            "image_id": [],
            "box_id": [],
            "gt": [],
            "match": [],
            "iou": [],
            "gt_class": [],
        }
        self.masks_dict = {
            "image_id": [],
            "mask_id": [],
            "gt": [],
            "match": [],
            "iou": [],
            "gt_class": [],
        }

        self.image_dict = {
            "image_id": [],
            "num_detections": [],
        }

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

        # batch metrics, mAP, classful, per class
        self.det_map_classful = MeanAveragePrecision(
            iou_type="bbox", average="macro", class_metrics=True, extended_summary=False
        )

        # calculating something at all, if det=None, then gt is used
        assert (model_seg is not None) or (
            model_det is not None
        ), "have to evaluate something: det, seg or both"

    def matching(self, gt, det, det_scores, iou_type):
        """
        returns matched objects and IoUs of the matched pairs
        threshold is hardcoded to 0.5 to filter bad matches
        """
        return matching.matching_fn(
            gt, det, det_scores, threshold=0.5, iou_type=iou_type
        )

    def store_gt(self, boxes, masks, classes, image_id):
        # add to dicts for saving later. gt masks/boxes and classes
        boxes = boxes.numpy()
        masks = masks.numpy()
        classes = classes.numpy()

        self.boxes_dict["gt"].extend(boxes)
        self.masks_dict["gt"].extend(masks)
        self.boxes_dict["gt_class"].extend(classes)
        self.masks_dict["gt_class"].extend(classes)

        # id of mask/box in the individual image, same len 0,1,2,3,...
        range_gt = np.arange(len(boxes)).tolist()
        self.boxes_dict["box_id"].extend(range_gt)
        self.masks_dict["mask_id"].extend(range_gt)

        # all these have the same image number 10,10,10,..., just store the image number
        img_idx_arr = np.ones(len(boxes), dtype=np.int32) * image_id
        self.boxes_dict["image_id"].extend(img_idx_arr)
        self.masks_dict["image_id"].extend(img_idx_arr)

    def store_matches(self, matches, ious, object_type):
        """
        One set of matches at a time
        Is stored inside a dict of lists for future reference when computing metrics or saving
        """

        if object_type == "boxes":
            self.boxes_dict["iou"].extend(ious)
            self.boxes_dict["match"].extend(matches)
        elif object_type == "masks":
            self.masks_dict["iou"].extend(ious)
            self.masks_dict["match"].extend(matches)

    def store_detections(self, indices, results):
        """
        Store the number of detections for every image,
        + save the detections themselves (boxes, masks) if needed
        Uses saver class passed down from pipeline, so evaluator does not need to know about paths and such.
        """
        if "masks" not in results.keys():  # dummy masks, no segmentation
            results["masks"] = [torch.tensor([]) for _ in range(len(results["boxes"]))]

        for image_id, boxes, masks, labels in zip(
            indices, results["boxes"], results["masks"], results["class_labels"]
        ):

            num_detections = len(boxes)
            self.image_dict["image_id"].append(image_id)
            self.image_dict["num_detections"].append(num_detections)

            # save the result dictionary into one file per image
            if self.cfg.save_results and self.cfg.save_results_per_image:
                # save anything and save per image
                self.saver.save_per_image(boxes, masks, labels, image_id)

    def prepare_gt(self, metadata, indices):
        # processes metadata of an image to access gt boxes, masks and classes,
        gt_boxes, gt_masks, gt_classes = [], [], []
        for instance, index in zip(metadata, indices):
            # extract from metadata
            boxes = instance["boxes"]
            masks = instance["masks"]
            classes = instance["categories"]

            # add to dicts for saving. gt masks/boxes and classes
            self.store_gt(boxes, masks, classes, index)

            # add to lists to return to eval for batch
            gt_boxes.append(boxes)
            gt_classes.append(classes)
            gt_masks.append(masks)

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

        # extract the results
        detected_objects = results[object_type]
        detected_classes = results["class_labels"]
        detected_scores = results["confidence"]

        # extract gt from its dict
        gt_objects = gt[object_type]
        gt_classes = gt["classes"]

        # convert to dicts for mAP class to calculate with.
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
            self.det_map_classful.update(preds=detected_dict, target=gt_dict)
        elif object_type == "masks":
            self.seg_map_classless.update(preds=detected_dict, target=gt_classless_dict)

        # 1:1 matching and IoU calculation for each image
        for batch_idx in range(len(detected_objects)):
            to_match_per_img = detected_objects[batch_idx].cpu()
            scores_per_img = detected_scores[batch_idx].cpu()
            gt_objects_per_img = gt_objects[batch_idx].cpu()

            # add inverse masks if needed
            if object_type == "masks" and self.cfg.add_inverse:

                if to_match_per_img.shape[0] != 0:  # if somethign detected
                    inverse = ~to_match_per_img
                    to_match_per_img = torch.concat((to_match_per_img, inverse), dim=0)
                    scores_per_img = torch.concat(
                        (scores_per_img, scores_per_img), dim=0
                    )

            # match 1:1 and store the matches.
            matched_detections, match_ious = self.matching(
                gt=gt_objects_per_img,  # tensor
                det=to_match_per_img,  # tensor
                det_scores=scores_per_img,  # tensor
                iou_type=object_type,
            )
            if self.cfg.save_matches_to_gt:
                self.store_matches(matched_detections, match_ious, object_type)

    def get_metrics(self):
        """
        Do final calculation of torchmetrics classes and wrap the results in a dict
        returns result dict, box and mask IoU arrays
        """

        # finally create dataframes from the dicts, filled with []
        self.masks_dict = pd.DataFrame(self.masks_dict)
        self.boxes_dict = pd.DataFrame(self.boxes_dict)
        self.image_dict = pd.DataFrame(self.image_dict)

        # average IoU
        det_iou = float(np.mean(np.array(self.boxes_dict["iou"])))
        seg_iou = float(np.mean(np.array(self.masks_dict["iou"])))

        # TP,FN,Recall etc.
        total_detections = self.image_dict["num_detections"].sum()
        detection_metrics = _compute_metrics_for_frame(
            self.boxes_dict, total_detections
        )
        segmentation_metrics = _compute_metrics_for_frame(
            self.masks_dict, total_detections
        )

        # add precomputed metrics to these dicts
        detection_metrics.update(
            {
                "avg iou": det_iou,
                "mAP without classes": utils.convert_tensors_to_save(
                    self.det_map_classless.compute()
                ),
                "mAP with classes": utils.convert_tensors_to_save(
                    self.det_map_classful.compute()
                ),
            }
        )

        segmentation_metrics.update(
            {
                "avg iou": seg_iou,
                "mAP without classes": utils.convert_tensors_to_save(
                    self.seg_map_classless.compute()
                ),
            }
        )

        return {
            "detection": detection_metrics,
            "segmentation": segmentation_metrics,
        }

    def get_results(self):
        """
        Packages the results into a dict. useful for saving via ResultSaver class
        Metrics are computed from stored gt and detection data via self.get_metrics()
        """
        return {
            "metrics": self.get_metrics(),
            "boxes_df": self.boxes_dict,  # dataframe containing matched boxes
            "masks_df": self.masks_dict,  # dtafram econtaining matched masks
            "image_level_df": self.image_dict,  # contains number of detections per image
        }

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
        for batch_idx, (images, metadata, indices) in tqdm(enumerate(data_loader)):
            # early stop
            if (max_batch is not None) and (batch_idx >= max_batch):
                break

            # list of list of mask/boxes and classes as GT
            # retriesve and save gt boxes, masks, classes
            gt = self.prepare_gt(metadata, indices)

            # detection module. attention points and labels possible with detection classes
            detection_results = self.model_det.detect_batch(
                images, metadata
            )  # need metadata for GT for some of them?

            # detection metrics
            self.update_metrics(results=detection_results, gt=gt, object_type="boxes")

            # can do box transforms here
            if self.boxes_transform is not None:
                detected_boxes = boxes_transform(detected_boxes)

            # segment, calculate metrics
            if self.model_seg is not None:
                segmentation_results = self.model_seg.infer_batch(
                    images, detection_results  # pass boxes and images
                )

                # same classes as for detection for metric computation
                segmentation_results["class_labels"] = detection_results["class_labels"]
                self.update_metrics(
                    results=segmentation_results, gt=gt, object_type="masks"
                )
                detection_results["masks"] = segmentation_results["masks"]

            # save the results per image
            self.store_detections(indices, detection_results)

            # cuda clean-up
            torch.cuda.empty_cache()


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
        saver=None,
        model_det=detector,
        model_seg=segmentation_model,
        boxes_transform=boxes_transform,
    )
    print("Evaluator test passed!")


if __name__ == "__main__":
    test_evaluator()
