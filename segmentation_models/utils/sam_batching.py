import torch
import numpy as np


def _prepare_image_for_batch(device, image, resize_transform):
    """
    Prepare the image for batch processing
    """
    image = resize_transform.apply_image(image)  # wants HWC (numpy) not CHW (torch)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()  # CHW


def _select_best_masks(batched_output, resulting_masks, index_list):
    # resulting masks contain [] in places where no boxes were detected
    # then return this array with new masks in the rest
    resulting_masks = resulting_masks.copy()
    for j, dict_output in enumerate(batched_output):
        pred_quality = dict_output["iou_predictions"]
        best = np.argmax(pred_quality.cpu(), axis=1)

        arange = torch.arange(best.shape[0])
        best_masks = dict_output["masks"][arange, best]
        resulting_masks[index_list[j]] = best_masks
    return resulting_masks
