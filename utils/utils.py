import numpy as np
import torch
from math import ceil as cl


def get_IoU_masks(gt_mask, mask):
    """
    Compute IoU between 2 masks
    """

    intersection = torch.sum(gt_mask and mask)
    union = torch.sum(gt_mask or mask)

    return 0 if union == 0 else (intersection / union)


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


def get_middle_point(box):
    """
    Get the middle point of a bounding box in format x0,y0,x1,y1
    """
    x0, y0 = box[0], box[1]
    x1, y1 = box[2], box[3]
    return [(x0 + x1) / 2, (y0 + y1) / 2]


def crop_xyxy(img, mask, box, crop_box):
    """
    Input:
        img: PIL image, ndarray image, or torch.Tensor (format CxHxW)
        mask: np.array mask
        box: bounding box in format x0,y0,x1,y1
        crop_box: crop box in format x0,y0,x1,y1
            x0,y0 is the left upper corner of the crop
            x1,y1 is the right lower corner of the crop

    Description:
    Crop the image and mask, starting at coordinates x0,y0 at the left upper corner.
    The resulting window has a size defined by the width and height of crop_box.
    """
    x0, y0, x1, y1 = cl(crop_box[0]), cl(crop_box[1]), cl(crop_box[2]), cl(crop_box[3])

    if img.__class__.__name__ == "Image":  # PIL image
        cropped_img = img.crop((x0, y0, x1, y1))
    elif img.__class__.__name__ == "ndarray":  # numpy array (most likely)
        cropped_img = img[y0:y1, x0:x1]
    elif img.__class__.__name__ == "Tensor":  # torch tensor
        cropped_img = img[y0:y1, x0:x1]
    else:
        raise ValueError("Unknown image type")

    box_coords = [
        box[0] - x0,
        box[1] - y0,
        box[2] - x0,
        box[3] - y0,
    ]  # subtract corner from the box
    cropped_mask = mask[y0:y1, x0:x1]  # need to also crop the w,h

    return cropped_img, cropped_mask, box_coords
