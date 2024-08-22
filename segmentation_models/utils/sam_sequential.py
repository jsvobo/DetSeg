import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import torch
import utils


def _select_best_mask(masks, scores):
    """
    Get the best mask from the list
    """
    idx = np.argmax(scores)
    return torch.Tensor(masks[idx]), scores[idx]


def _infer_masks_single_image(
    image: torch.Tensor,
    sam_predictor: SamPredictor,
    boxes: list = None,
    point_coords: list = None,
    point_labels: list = None,
):
    """
    Generates masks for a single image based on the provided prompts.
    Args:
        image (torch.Tensor): The input image.
        boxes (list, optional): List of bounding boxes. Defaults to None.
        point_coords (list, optional): List of point coordinates. Defaults to None.
        point_labels (list, optional): List of point labels. Defaults to None.
    Returns:
        dict: A dictionary containing the inferred masks and their scores.
            - "masks" (list): List of inferred masks.
            - "scores" (list): List of mask scores.

    """
    has_labels = point_labels is not None
    has_points = point_coords is not None
    has_boxes = boxes is not None

    # input checks
    assert has_boxes or has_points
    if has_points:
        assert has_labels

    # how many prompt sets are there??
    if has_boxes:
        len_prompts = len(boxes)
        if len_prompts == 0:  # boxes, but empty (some images don't have boxes)
            return {"masks": [], "scores": []}
    else:
        len_prompts = len(point_coords)

    inferred_masks = []
    mask_scores = []
    sam_predictor.set_image(np.array(image))

    for i in range(len_prompts):
        points = point_coords[i] if has_points else None
        labels = point_labels[i] if has_labels else None
        box = boxes[i] if has_boxes else None

        # run inferrence on one prompt and return the best mask
        masks, scores, logits = sam_predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=np.array(box),
            multimask_output=True,
        )
        best_mask, best_score = _select_best_mask(masks, scores)
        inferred_masks.append(best_mask)
        mask_scores.append(best_score)

    return {"masks": inferred_masks, "scores": mask_scores}
