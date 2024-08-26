import numpy as np
import torch
from math import ceil as cl


def get_IoU_masks(gt_mask, mask):
    """
    Compute IoU between 2 masks
    """

    intersection = torch.sum(torch.logical_and(gt_mask, mask))
    union = torch.sum(torch.logical_or(gt_mask, mask))

    return 0 if union == 0 else (intersection / union)


def get_IoU_multiple(masks, gt_masks):
    """
    Compute IoU between 2 masks
    """
    assert len(gt_masks) == len(masks)
    return [get_IoU_masks(gt_masks[i], masks[i]) for i in range(len(masks))]


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
    x0, y0, x1, y1 = crop_box

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


def convert_tensors_to_save(d):
    if isinstance(d, dict):
        # Recursively apply the function for nested dictionaries
        return {k: convert_tensors_to_save(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        # Convert the torch.Tensor to a numpy array
        return d.cpu().tolist()
    else:
        # Return the value as is if it's neither a dict nor a torch.Tensor
        return d
