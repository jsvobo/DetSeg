import numpy as np
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import torch
import utils
import segmentation_models.utils.sam_sequential
import segmentation_models.utils.sam_batching


class SamWrapper:

    def __init__(self, device="cuda", model="b"):
        assert device in ["cpu", "cuda"]
        assert model in ["b", "h"]

        self.device = device
        self.sam_predictor, self.sam = self.prepare_sam(model=model)

    def infer_masks_sequential(self, items, boxes=None, points=None, point_labels=None):
        """
        Does not use batching, sequentially processes the images, returns list of dicts
            Each dict has masks and their scores for one image

        Args:
            items (list): List of images to process
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
            box = boxes[i] if has_boxes else item["annotations"]["boxes"]

            results = sam_sequential._infer_masks_single_image(  # segment the image
                image=item["image"],
                sam_predictor=self.sam_predictor,
                boxes=box,
                point_coords=points,
                point_labels=labels,
            )
            results_whole.append(results)
        return results_whole

    def infer_masks_batch(
        self, batch, boxes, point_coords=None, point_labels=None, metrics_class=None
    ):
        # batch inferrence
        pass

    def prepare_sam(self, model):
        """
        Load SAM-1 predictor and model itself (needed for some functions)
        """
        # I know, I know, conf and all that, dynamic loading by different users etc... For now hardcoded
        # sam_vit_b_01ec64.pth
        # vit_b
        # sam_vit_h_4b8939.pth
        # vit_h

        if model == "b":  # TODO: dict and not like this
            sam_checkpoint = "/mnt/vrg2/imdec/models/sam1/sam_vit_b_01ec64.pth"
            model_type = "vit_b"

        elif model == "h":
            sam_checkpoint = "/mnt/vrg2/imdec/models/sam1/sam_vit_h_4b8939.pth"
            model_type = "vit_h"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)

        return SamPredictor(sam), sam


class AutomaticSam(SamAutomaticMaskGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def automatic_one_image(self, image, mask_generator):
        image = np.array(utils.to_plt_order(image))
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


def test_sam_wrapper():
    pass
