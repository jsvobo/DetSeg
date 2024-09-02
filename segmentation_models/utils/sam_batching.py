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
    resulting_masks = resulting_masks.copy()  # will be [] from filtering beforehand
    iou_preds = []
    for idx_in_batch, dict_output in enumerate(batched_output):
        pred_qualities = dict_output["iou_predictions"].cpu()
        best = np.argmax(
            pred_qualities, axis=1
        )  # find which mask is the best for each box

        # take this mas and its quality
        arange = torch.arange(best.shape[0])
        iou_preds.append(pred_qualities[arange, best])
        best_masks = dict_output["masks"][arange, best]

        resulting_masks[index_list[idx_in_batch]] = (
            best_masks  # add mask where there is any
        )
    return {"masks": resulting_masks, "confidence": iou_preds}
