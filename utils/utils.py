import numpy as np
import torch


def area(box):
    x0, y0, x1, y1 = box
    return (x1 - x0) * (y1 - y0)


def get_IoU_boxes(box1, box2):
    """
    Compute IoU between 2 bounding boxes
    """
    x0, y0, x1, y1 = box1
    x0_, y0_, x1_, y1_ = box2

    if x0_ > x1 or y0_ > y1 or x1_ < x0 or y1_ < y0:
        # out completely, no intersection
        return 0

    max_left_x = max(x0, x0_)
    max_upper_y = max(y0, y0_)
    min_right_x = min(x1, x1_)
    min_lower_y = min(y1, y1_)

    area1 = int(area(box1))
    area2 = int(area(box2))

    intersection_box = [max_left_x, max_upper_y, min_right_x, min_lower_y]
    intersection = area(intersection_box)
    union = area1 + area2 - intersection

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
    Compute IoU between 2 sets of MATCHED masks (1:1),
        otherwise the result is bad. no matching here
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
