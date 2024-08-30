import numpy as np
import torch
import utils
from omegaconf import DictConfig, OmegaConf

import segmentation_models.utils.sam_sequential as sam_sequential
import segmentation_models.utils.sam_batching as sam_batching
from segmentation_models.base_seg_wrapper import BaseSegmentWrapper

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import yaml


def prepare_sam(device, model, cfg=None):
    """
    Load SAM-1 predictor and model itself, before passing them to wrappers for example
    config: model_type and path to checkpoint
    """
    config_path = (
        cfg.sam_path_config
        if cfg is not None
        else "./config/segmentation/other/sam1_paths.yaml"
    )
    with open(config_path, "r") as f:
        subconfig = yaml.safe_load(f)[model]
    sam_checkpoint = subconfig["path"]
    model_type = subconfig["model_type"]

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return SamPredictor(sam), sam


class SamWrapper(BaseSegmentWrapper):
    """
    Does not extend sam class, but uses it to infer masks and generate automatic masks
    Can have other segmentation models. Important functions are outlined in the base class
    Use infer_masks to get masks for a list of single images or infer_batch for inferrence of multiple prompts and multiple images.
    infer_masks is sequential in a sense, to be used for visualisations.
    """

    def __init__(self, device="cuda", model="b", cfg: DictConfig = None):
        assert device in ["cpu", "cuda"]
        self.device = device
        self.sam_predictor, self.sam = prepare_sam(device=device, cfg=cfg, model=model)
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
        for image_idx in range(len(images)):
            img = images[image_idx]
            boxes_for_image = boxes[image_idx]
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
                coords = torch.Tensor([point_coords[image_idx]])
                points_for_image = coords.permute(1, 0, 2).to(self.device)
                labels = torch.Tensor([point_labels[image_idx]])
                labels_for_image = labels.T.to(self.device)

                # add to dict
                dict_img["point_coords"] = self.resize_transform.apply_coords_torch(
                    points_for_image, original_size
                )
                dict_img["point_labels"] = labels_for_image

            sam_batched_inputs.append(dict_img)
            index_list.append(image_idx)

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

    def automatic_one_image(self, image):
        masks_dicts = self.generate(image)

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
        """
        inputs:
            list of items, each item is a dict with key "image" containing the image
        outputs:
            list of dicts, each dict contains the keys "boxes", "masks", "points"

        example in sam.ipynb
        """
        generated_masks_dicts = []
        for item in items:
            image = item["image"]
            detected_boxes, detected_masks, points = self.automatic_one_image(image)
            generated_masks_dicts.append(
                {"boxes": detected_boxes, "masks": detected_masks, "points": points}
            )
        return generated_masks_dicts


def test_sam1_wrappers():
    print("\nTesting sam1 wrapper:")

    sam_wrapper = SamWrapper(
        device="cuda", model="b"
    )  # without config now? just want default b model
    image = np.array(torch.randn(480, 640, 3))
    box = torch.tensor([[0, 0, 100, 100]])

    # infer masks, same box and random image, but once adding box directly and once as a gt
    masks1 = sam_wrapper.infer_masks(items=[{"image": image}], boxes=[[box]])
    masks2 = sam_wrapper.infer_masks(
        items=[
            {"image": image, "annotations": {"boxes": [box]}}
        ]  # simulate our data structure
    )
    assert torch.eq(
        masks1[0]["masks"][0], masks2[0]["masks"][0]
    ).all()  # the output is the same

    print("Sam1 wrapper test passed! Commencing with automatic mask generator testing")

    sam_auto = AutomaticSam(
        model=sam_wrapper.get_sam(),  # take sam loaded from wrapper
        points_per_side=32,
        pred_iou_thresh=0.9,  # rejects "bad" masks
        stability_score_thresh=0.95,  # rejects "bad" masks
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=200,
    )

    automatic_segmentations = sam_auto.automatic_multiple_images(
        items=[{"image": image}]
    )  # segment automatically using the generator above

    print("Automatic segmentation test passed!")


if __name__ == "__main__":
    test_sam1_wrappers()
