import numpy as np
import torch
import utils

import segmentation_models.utils.sam_sequential as sam_sequential
import segmentation_models.utils.sam_batching as sam_batching
from segmentation_models.base_seg_wrapper import BaseSegmentWrapper

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator


class SamWrapper(BaseSegmentWrapper):

    def __init__(self, device="cuda", model="b"):
        assert device in ["cpu", "cuda"]
        assert model in ["b", "h"]

        self.device = device
        self.sam_predictor, self.sam = prepare_sam(model=model, device=device)
        self.resize_transform = ResizeLongestSide(self.get_image_size())

    def infer_masks(self, items, boxes=None, points=None, point_labels=None):
        """
        Does not use batching, sequentially processes the images, returns list of dicts
            Each dict has masks and their scores for one image

        Args:
            items (list): List of dicts with images and GT annotations
            boxes (list): List of bounding boxes for prompting, Optional, then use GT boxes
            points (list, optional): List of point coordinates for prompting. Defaults to None.
            point_labels (list, optional): List of point labels for prompting. Defaults to None.

        Returns:
            dict: A dictionary containing the inferred masks and their scores.
                - "masks" (list): List of inferred masks.
                - "scores" (list): List of mask scores.
        """
        has_labels = point_labels is not None
        has_points = points is not None
        has_boxes = boxes is not None

        results_whole = []
        for i, item in enumerate(items):

            points = points[i] if has_points else None
            labels = point_labels[i] if has_labels else None
            boxes_image = boxes[i] if has_boxes else item["annotations"]["boxes"]

            results = sam_sequential._infer_masks_single_image(  # segment the image
                image=item["image"],
                sam_predictor=self.sam_predictor,
                boxes=boxes_image,
                point_coords=points,
                point_labels=labels,
            )
            results_whole.append(results)
        return results_whole

    def infer_batch(
        self,
        images,
        boxes=None,
        point_coords=None,
        point_labels=None,
    ):
        sam_batched_inputs = []
        resulting_masks = []
        index_list = []

        # prepare sam_batched_inputs
        for j in range(len(images)):
            img = images[j]
            boxes_for_image = boxes[j]
            resulting_masks.append(torch.Tensor([]))

            if len(boxes_for_image) == 0:
                continue  # no boxes in image

            dict_img = {  # written according to official sam notebook predictor.ipynb
                "image": sam_batching._prepare_image_for_batch(
                    image=img,
                    resize_transform=self.resize_transform,
                    device=self.sam.device,
                ),
                "boxes": self.resize_transform.apply_boxes_torch(
                    boxes_for_image.to(self.sam.device), img.shape[:2]
                ),
                "original_size": img.shape[:2],
            }
            sam_batched_inputs.append(dict_img)
            index_list.append(j)

        # batch inference
        if sam_batched_inputs == []:
            return resulting_masks  # [[], [], ...]

        batched_output = self.sam(sam_batched_inputs, multimask_output=True)

        # dict_keys(['masks', 'iou_predictions', 'low_res_logits'])
        for j, dict_output in enumerate(batched_output):
            pred_quality = dict_output["iou_predictions"]
            best = np.argmax(pred_quality.cpu(), axis=1)

            arange = torch.arange(best.shape[0])
            best_masks = dict_output["masks"][arange, best]
            resulting_masks[index_list[j]] = best_masks

            # use index_list to map back to original image ordereven with missing images

        return resulting_masks


class AutomaticSam(SamAutomaticMaskGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def automatic_one_image(self, image, mask_generator):
        masks_dicts = mask_generator.generate(image)

        detected_boxes = []
        detected_masks = []
        points = []
        for mask in masks_dicts:  # prompt by box only
            detected_boxes.append(mask["bbox"])
            detected_masks.append(mask["segmentation"])
            points.append(mask["point_coords"])

        detected_masks = np.array(detected_masks)
        detected_boxes = np.array(detected_boxes)
        points = np.array(points).T[:, 0]

        return detected_boxes, detected_masks, points  # for each image

    def automatic_multiple_images(self, items, mask_generator):
        generated_masks_dicts = []
        for item in items:
            image = item["image"]
            detected_boxes, detected_masks, points = self.automatic_one_image(
                image, mask_generator
            )
            generated_masks_dicts.append(
                {"boxes": detected_boxes, "masks": detected_masks, "points": points}
            )
        return generated_masks_dicts


def prepare_sam(model, device):
    """
    Load SAM-1 predictor and model itself
    """
    # TODO: config file
    # sam_vit_b_01ec64.pth
    # vit_b
    # sam_vit_h_4b8939.pth
    # vit_h

    if model == "b":
        sam_checkpoint = "/mnt/vrg2/imdec/models/sam1/sam_vit_b_01ec64.pth"
        model_type = "vit_b"

    elif model == "h":
        sam_checkpoint = "/mnt/vrg2/imdec/models/sam1/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return SamPredictor(sam), sam


def test_sam_wrappers():
    print("Not implemented yet")
    # load wrapper and the automatic generator wrapper
    # do some very basic testing


if __name__ == "__main__":
    test_sam_wrappers()
