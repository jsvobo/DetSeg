from torchmetrics import MetricCollection
import torch
import torchvision
from torchmetrics.classification import JaccardIndex
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np

import utils  # contains sam_utils, visual_utils, and other utility functions
from datasets.dataset_loading import CocoLoader, get_coco_split
import segmentation_models


def masks_to_dict_jaccard(boxes):
    """
    input is a list of boxes for one image
    output is a list of dictionaries with each box in one
    """
    return [
        {"boxes": box, "labels": [0]} for box in boxes
    ]  # all the same class for now?


def boxes_to_dict_map(list_of_list_of_boxes):
    """
    input is a list of lists boxes for one image
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
    return [
        {
            "boxes": list_of_masks,
            "labels": torch.zeros(len(list_of_masks), dtype=torch.int32),
            "scores": torch.ones(len(list_of_masks), dtype=torch.float32),
        }
        for list_of_masks in list_of_list_of_masks
    ]


class DummyDetectorBase:
    def __init__(self):
        pass

    def extract_GT_boxes(self, metadata_list):
        results = [torch.Tensor(instance["boxes"]) for instance in metadata_list]
        return results  # returns list of lists of boxes from metadata

    def detect_batch(self, images, metadata=None, classes=None):
        raise NotImplementedError


class GTboxDetector(DummyDetectorBase):
    """
    Dummy detector class, which just takes GT bboxes with no other prompts
    """

    def detect_batch(self, images, metadata=None, classes=None):
        return self.extract_GT_boxes(metadata), None, None
        # gives no point labels or attention points, simply GT boxes


class GTboxDetectorMiddle(DummyDetectorBase):
    """
    Dummy detector class, which (naively) takes the middle point as a positive prompt to sam
    """

    def detect_batch(self, images, metadata=None, classes=None):
        results = self.extract_GT_boxes(metadata)  # list of lists of boxes

        points = []
        prompts = []
        for boxes in results:  # not batching, sequentially load and middles
            points.append([utils.get_middle_point(box) for box in boxes])  # [[x,y],...]
            prompts.append([[1] for box in boxes])  # [[1],[1],...]
        return results, np.array(points), np.array(prompts)


def dummy_box_matching(gt_boxes_list, detected_boxes_list):
    """
    Returns a list of boxes which are "best" for each instance of GT boxes
    For purposes of metrics.
    More interested in recall and
    """
    return [
        detected_boxes_list[i][
            : len(gt_boxes)
        ]  # for each sublist takes the same number of boxes as is GT (in order, just dummy)
        for i, gt_boxes in enumerate(gt_boxes_list)
    ]  # just return some boxes
    # in eality, detected can have fewer boxes, then what?


class Evaluator:
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
        self.seg_batch_metrics = seg_batch_metrics
        self.seg_pairwise_metrics = seg_pairwise_metrics
        self.boxes_transform = boxes_transform
        self.box_matching = box_matching
        self.device = device

        # calculating something at all
        assert (model_seg is not None) or (model_det is not None)

        # we want some metrics from segmentation, if we have a model there, otherwise we can skip
        if model_seg is not None:
            assert (seg_pairwise_metrics is not None) or (seg_batch_metrics is not None)
            # can have detection without detection metrics tho

        if model_det is None:
            self.model_det = GTboxDetector()
            # takes GT bboxes, need something for segmentation

    def evaluate(self, data_loader, savepath=None, max_batch=None):
        # TODO: append IoU somewhere, boxes, masks, metadata, save into file?
        classes = (
            data_loader.dataset.get_classes()
        )  # if classes are needed for detection (DINO etc.)

        for i, batch in enumerate(data_loader):
            if (max_batch is not None) and (i > max_batch):
                break
            images = list(batch[0])
            metadata = list(batch[1])

            # list of list of mask/boxes as GT
            gt_boxes = [instance["boxes"] for instance in metadata]
            gt_masks = [instance["masks"] for instance in metadata]

            detected_boxes, attention_points, point_labels = (
                self.model_det.detect_batch(images, metadata, classes)
            )

            # match boxes to GT (right now 1:1, but can be more complex)
            selected_boxes = self.box_matching(
                gt_boxes_list=gt_boxes, detected_boxes_list=detected_boxes
            )

            if self.det_batch_metrics is not None:
                self.det_batch_metrics.update(
                    boxes_to_dict_map(selected_boxes),
                    boxes_to_dict_map(gt_boxes),
                )  # on the batch

            if boxes_transform is not None:
                detected_boxes = boxes_transform(
                    detected_boxes
                )  # for example slightly larger boxes? default=None

            # if seg, then some metrics are present, calculate per instance

            if self.model_seg is not None:
                inferred_masks = self.model_seg.infer_batch(
                    images, selected_boxes, attention_points, point_labels
                )

                if self.seg_batch_metrics is not None:
                    # calculate metrics (like map) over batches
                    self.seg_batch_metrics.update(
                        masks_to_dict_map(inferred_masks),
                        masks_to_dict_map(gt_masks),
                    )

                if self.seg_pairwise_metrics is not None:
                    for j, (gt_mask, inferred_mask) in enumerate(
                        zip(gt_masks, inferred_masks)
                    ):
                        for k, (gt, inferred) in enumerate(zip(gt_mask, inferred_mask)):
                            self.seg_pairwise_metrics.update(inferred, gt)

        # TODO: save something needed? parametrise what?


if __name__ == "__main__":

    #  from config here?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batchsize = 6  # h4, b6 on RTX4000, b8,h6 on A100
    sam_model = "b"  # base or huge

    # if None, then GTboxDetector is initialised inside eval for GT anyways
    detector = GTboxDetector()

    dataset_transforms = None
    coco_val_dataset = CocoLoader(  # load coco
        get_coco_split(split="val"), transform=dataset_transforms
    )
    # box matching (select from detected boxes to then compare to GT in metrics)
    box_matching = dummy_box_matching
    # sam
    model_seg = None  # segmentation_models.SamWrapper(device=device, model=sam_model)
    boxes_transform = None  # no transformation for now

    # metric collections
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
    seg_pairwise_metrics = MetricCollection(JaccardIndex("binary"))
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
        det_batch_metrics=det_batch_metrics,
        seg_batch_metrics=seg_batch_metrics,
        seg_pairwise_metrics=seg_pairwise_metrics,
        boxes_transform=boxes_transform,
        model_seg=model_seg,
        box_matching=box_matching,
    )

    # loader from dataset
    data_loader = coco_val_dataset.instantiate_loader(
        batch_size=batchsize, num_workers=4
    )

    # Run the evaluation
    evaluator.evaluate(data_loader, max_batch=10)

    print(det_batch_metrics.compute())
    if model_seg is not None:
        print(seg_batch_metrics.compute())
        print(seg_pairwise_metrics.compute())

    # process results further (mAP, Recall, IoU(s) etc.)
