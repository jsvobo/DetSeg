import numpy as np
import torch


def get_coco_boxes(metadata):
    """
    Takes the COCO metadata and returns the bounding boxes.
    for one image
    """
    boxes = []
    for ann in metadata:
        box = box_coco_to_sam(ann["bbox"])
        boxes.append(box)

    return boxes


def get_coco_masks(metadata: dict, api_class):
    """
    Takes the COCO metadata and returns the segmentation masks.
    for one images
    """
    masks = []
    for ann in metadata:
        mask = api_class.annToMask(ann)
        masks.append(mask)

    return masks


def coco_masks_boxes(metadata, api_class):
    """
    Takes the COCO metadata and returns the segmentation masks and boxes in one run-through.
    for one images
    """
    boxes = []
    masks = []
    for ann in metadata:
        box = box_coco_to_sam(ann["bbox"])
        boxes.append(box)
        mask = api_class.annToMask(ann)
        masks.append(mask)

    return np.array(masks), boxes


def get_IoU_masks(gt_mask, mask):
    """
    Compute IoU between 2 masks
    """
    if type(gt_mask) == np.ndarray:
        library = np
    else:
        library = torch
    intersection = library.sum((1 * gt_mask + 1 * mask) == 2)
    union = library.sum((1 * gt_mask + 1 * mask) != 0)
    IoU = intersection / union

    return IoU


def get_IoU_multiple(masks, gt_masks):
    """
    Compute IoU between 2 masks
    """
    assert len(gt_masks) == len(masks)
    return [get_IoU_masks(gt_masks[i], masks[i]) for i in range(len(masks))]


def box_coco_to_sam(coco_box):
    """
    Convert coco box to sam box
    from x0,y0,w,h to x0,y0,x1,y1
    """
    return (
        coco_box[0],
        coco_box[1],
        coco_box[0] + coco_box[2],
        coco_box[1] + coco_box[3],
    )


def boxes_coco_to_sam(coco_boxes):
    """
    Convert coco boxes to sam boxes
    from x0,y0,w,h to x0,y0,x1,y1
    """
    sam_boxes = []
    for box in coco_boxes:
        sam_boxes.append(box_coco_to_sam(box))
    return sam_boxes


def get_middle_point(box):
    """
    Get the middle point of a bounding box in format x0,y0,x1,y1
    """
    x0, y0 = box[0], box[1]
    x1, y1 = box[2], box[3]
    return [(x0 + x1) / 2, (y0 + y1) / 2]
