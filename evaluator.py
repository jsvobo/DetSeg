from torchmetrics import MetricCollection
import torch
import torchvision
from torchmetrics.classification import JaccardIndex
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
from tqdm import tqdm

import utils  # contains sam_utils, visual_utils, and other utility functions
from datasets.dataset_loading import CocoLoader, get_coco_split
import segmentation_models
import detection_models


# helper functions for evaluator list of lists of boxs/masks -> list of dicts with boxes/masks
def boxes_to_dict_map(list_of_list_of_boxes):
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
    return [
        {
            "boxes": list_of_boxes,
            "labels": torch.zeros(len(list_of_boxes), dtype=torch.int32),
            "scores": torch.ones(len(list_of_boxes), dtype=torch.float32),
        }
        for list_of_boxes in list_of_list_of_boxes
    ]


def masks_to_dict_map(list_of_list_of_masks):
    """
    Converts a list of lists of masks to a list of dictionaries.
    Args:
        list_of_list_of_masks (list): A list of lists of masks.
    Returns:
        list: A list of dictionaries, where each dictionary contains the following keys:
            - "masks": The masks converted to CPU.
            - "labels": A tensor of zeros, since every object os of the same class for basic IoU.
            - "scores": A tensor of ones with the length of the masks.
    """

    return [
        {
            "masks": list_of_masks.to("cpu"),
            "labels": torch.zeros(len(list_of_masks), dtype=torch.int32),
            "scores": torch.ones(len(list_of_masks), dtype=torch.float32),
        }
        for list_of_masks in list_of_list_of_masks
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
        self.det_batch_metrics = det_batch_metrics
        self.seg_batch_metrics = seg_batch_metrics  # mAP, can be run on the batch
        self.seg_pairwise_metrics = (
            seg_pairwise_metrics  # matrix on mask-mask basis (Jaccard)
        )
        self.boxes_transform = boxes_transform
        self.box_matching = box_matching
        self.device = device

        # calculating something at all
        assert (model_seg is not None) or (model_det is not None)

        # we want some metrics from segmentation, if we have a model there, otherwise we can skip
        if model_seg is not None:
            assert (seg_pairwise_metrics is not None) or (seg_batch_metrics is not None)

        # takes GT bboxes, need something for segmentation still
        if model_det is None:
            self.model_det = detection_models.GTboxDetector()

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
        # TODO: append IoU somewhere, boxes, masks, metadata, save into file?

        # if classes are needed for detection (DINO etc.)
        classes = data_loader.dataset.get_classes()

        for i, batch in tqdm(enumerate(data_loader)):
            if (max_batch is not None) and (i > max_batch):
                break
            images = list(batch[0])
            metadata = list(batch[1])

            # list of list of mask/boxes as GT
            gt_boxes = [instance["boxes"] for instance in metadata]
            gt_masks = [instance["masks"].type(torch.uint8) for instance in metadata]

            # detection
            detected_boxes, attention_points, point_labels = (
                self.model_det.detect_batch(images, metadata, classes)
            )

            # select boxes for segmentation
            if self.box_matching is not None:
                selected_boxes = self.box_matching(
                    gt_boxes_list=gt_boxes, detected_boxes_list=detected_boxes
                )
            else:  # default
                selected_boxes = gt_boxes

            # batch detection metrics
            if self.det_batch_metrics is not None:
                self.det_batch_metrics.update(
                    boxes_to_dict_map(selected_boxes),
                    boxes_to_dict_map(gt_boxes),
                )

            # can do box transforms here
            if boxes_transform is not None:
                detected_boxes = boxes_transform(detected_boxes)

            # segment, calculater batch and then pairwise metrics
            if self.model_seg is not None:
                inferred_masks = self.model_seg.infer_batch(
                    images, selected_boxes, attention_points, point_labels
                )

                # batch metrics
                if self.seg_batch_metrics is not None:
                    self.seg_batch_metrics.update(
                        masks_to_dict_map(inferred_masks),
                        masks_to_dict_map(gt_masks),
                    )

                # pairwise  mask metrics
                if self.seg_pairwise_metrics is not None:
                    for j, (gt_masks_for_image, inferred_masks_for_image) in enumerate(
                        zip(gt_masks, inferred_masks)
                    ):
                        for k, (gt, inferred) in enumerate(
                            zip(
                                gt_masks_for_image.to(self.device),
                                inferred_masks_for_image.to(self.device),
                            )
                        ):
                            self.seg_pairwise_metrics.update(inferred, gt)

        # TODO: save something needed? individual IoUs, metadata, some images, idices?


def test_evaluator():
    print("Not implemented yet")


if __name__ == "__main__":
    #  from config here? loading from args?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batchsize = 6  # h4, b6 on RTX4000, b8,h6 on A100
    sam_model = "b"  # b,h = base or huge
    split = "val"  # train or val
    max_batch = 10  # early cutoff for testing

    detector = detection_models.GTboxDetector()
    # or detection_models.GTboxDetectorMiddle()

    # load coco dataset
    coco_val_dataset = CocoLoader(get_coco_split(split=split), transform=None)

    # load sam
    segmentation_model = segmentation_models.SamWrapper(device=device, model=sam_model)

    # additional functions that in the pipeline
    boxes_transform = None  # no transformation for now
    box_matching = None  # selects boxes to use for seg.

    # metrics for det, seg batch and seg pairwise
    det_batch_metrics = MetricCollection(
        [
            # IntersectionOverUnion(),
            MeanAveragePrecision(
                iou_type="bbox",
                average="macro",
                class_metrics=False,
            ),
        ]
    )
    seg_pairwise_metrics = MetricCollection(JaccardIndex("binary").to(device))
    seg_batch_metrics = MetricCollection(
        [
            # JaccardIndex("binary"),
            MeanAveragePrecision(
                iou_type="segm",
                average="macro",
                class_metrics=False,
            ),
        ]
    )

    # Instantiate the Evaluator with the requested parameters
    evaluator = Evaluator(
        device=device,
        model_det=detector,
        model_seg=segmentation_model,
        det_batch_metrics=det_batch_metrics,
        seg_batch_metrics=seg_batch_metrics,
        seg_pairwise_metrics=seg_pairwise_metrics,
        boxes_transform=boxes_transform,
        box_matching=box_matching,
    )

    # loader from dataset
    data_loader = coco_val_dataset.instantiate_loader(
        batch_size=batchsize, num_workers=4
    )

    # run the evaluation
    evaluator.evaluate(data_loader, max_batch=max_batch)

    # print results
    print("\n\n")
    print(det_batch_metrics.compute())
    print("\n")
    if segmentation_model is not None:
        print(seg_batch_metrics.compute())
        print("\n")
        print(seg_pairwise_metrics.compute())

    # process results further (mAP, Recall, IoU(s) etc.)
    # save results to file?

    """ 
    print("Mean IoU: " + str(dataset_IoU.compute()))
    filename = f"./out/coco_base_thresholds_{len(thresholds)}.npy"
    # save to a file
    with open(filename, "wb") as f:
        np.save(f, np.array(thresholds))
    """
