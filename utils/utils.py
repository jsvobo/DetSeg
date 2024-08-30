import numpy as np
import torch


def boxes_to_masks(boxes, mask_shape):
    """
    Convert list of boxes in formax x0,y0,x1,y1 to uint8 mask for calculating IoU. need to know the full shape
    """
    masks = []
    for box in boxes:
        mask = torch.zeros(mask_shape, dtype=torch.uint8)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 1
        masks.append(mask)
    return masks


def get_IoU_boxes(gt_box, det_box):
    """
    Compute IoU between 2 bounding boxes
    """
    x0, y0, x1, y1 = gt_box
    x0_, y0_, x1_, y1_ = det_box
    intersection = max(0, min(x1, x1_) - max(x0, x0_)) * max(
        0, min(y1, y1_) - max(y0, y0_)
    )
    area_gt = (x1 - x0) * (y1 - y0)
    area_det = (x1_ - x0_) * (y1_ - y0_)
    union = area_gt + area_det - intersection
    return 0 if union == 0 else (intersection / union)


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
    """
    Recursively convert dictionary with tensors to dictionary with lists at the leaves.
    This is done for saving purposes, as torch.Tensor cannot be saved to disk
    """
    if isinstance(d, dict):
        # Recursively apply the function for nested dictionaries
        return {k: convert_tensors_to_save(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        # Convert the torch.Tensor to a numpy array
        return d.cpu().tolist()
    else:
        # Return the value as is if it's neither a dict nor a torch.Tensor
        return d
