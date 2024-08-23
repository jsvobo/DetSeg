import numpy as np
import torch
import utils

import segmentation_models.utils.sam_sequential as sam_sequential
import segmentation_models.utils.sam_batching as sam_batching
from segmentation_models.base_seg_wrapper import BaseSegmentWrapper

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import yaml


class SamWrapper(BaseSegmentWrapper):

    def __init__(self, device="cuda", model="b"):
        assert device in ["cpu", "cuda"]
        assert model in ["b", "h"]

        self.device = device
        self.sam_predictor, self.sam = prepare_sam(model=model, device=device)
        self.resize_transform = ResizeLongestSide(self.get_image_size())

    def infer_masks(self, items, boxes=None, points=None, point_labels=None):

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

        # prepare sam_batched_inputs based on prompts
        for j in range(len(images)):
            img = images[j]
            boxes_for_image = boxes[j]
            resulting_masks.append(torch.Tensor([]))

            if len(boxes_for_image) == 0:
                continue  # no boxes in image

            original_size = img.shape[:2]
            dict_img = {  # written according to official sam notebook predictor.ipynb
                "image": sam_batching._prepare_image_for_batch(
                    image=img,
                    resize_transform=self.resize_transform,
                    device=self.device,
                ),
                "boxes": self.resize_transform.apply_boxes_torch(
                    boxes_for_image.to(self.device), original_size
                ),
                "original_size": original_size,
            }

            if point_coords is not None:
                # reshape to correct format BxNx2 and BxN
                coords = torch.Tensor([point_coords[j]])
                points_for_image = coords.permute(1, 0, 2).to(self.device)
                labels = torch.Tensor([point_labels[j]])
                labels_for_image = labels.T.to(self.device)

                # add to dict
                dict_img["point_coords"] = self.resize_transform.apply_coords_torch(
                    points_for_image, original_size
                )
                dict_img["point_labels"] = labels_for_image

            sam_batched_inputs.append(dict_img)
            index_list.append(j)

        # None came through
        if sam_batched_inputs == []:
            return resulting_masks  # [[], [], ...] at this point

        # batch inference
        batched_output = self.sam(sam_batched_inputs, multimask_output=True)
        results = sam_batching._select_best_masks(
            batched_output, resulting_masks, index_list
        )

        return results


class AutomaticSam(SamAutomaticMaskGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def automatic_one_image(self, image, mask_generator):
        masks_dicts = self.mask_generator.generate(image)

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

    def automatic_multiple_images(self, items):
        generated_masks_dicts = []
        for item in items:
            image = item["image"]
            detected_boxes, detected_masks, points = self.automatic_one_image(image)
            generated_masks_dicts.append(
                {"boxes": detected_boxes, "masks": detected_masks, "points": points}
            )
        return generated_masks_dicts


def prepare_sam(
    model,
    device,
):
    """
    Load SAM-1 predictor and model itself
    """
    assert model in ["b", "h"]

    with open("segmentation_models/config/sam1_conf.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_dict = config[model]

    sam_checkpoint = model_dict["path"]
    model_type = model_dict["model_type"]

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return SamPredictor(sam), sam


def test_sam_wrappers():
    print("Not implemented yet")
    # load wrapper and the automatic generator wrapper
    # do some very basic testing


if __name__ == "__main__":
    test_sam_wrappers()
