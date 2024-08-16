import torch
import os
import numpy as np
import torchvision.datasets
from torchvision.datasets import CocoDetection
from torchvision.transforms.functional import pil_to_tensor
import pycocotools


import utils  # otherwise does not work form main repo


def get_coco_split(split: str = "train", year: str = "2017"):
    """
    Returns the paths of the COCO dataset split for a given year.
    Args:
        split (str, optional): The split of the dataset. Defaults to "train".
        year (str, optional): The year of the dataset. Defaults to "2017".
    Returns:
        tuple: A tuple containing the image path and annotation path.
    """

    assert split in ["train", "val"]
    assert year in ["2014", "2017"]

    root = "./datasets/COCO"  # load from conf??
    annotation_path = os.path.join(root, "annotations")
    type_ann = "instances"  # instances, captions, person_keypoints

    ann_path = os.path.join(annotation_path, type_ann + "_" + split + year + ".json")
    image_path = os.path.join(root, split + year)
    return (image_path, ann_path)


class CocoLoader(CocoDetection):

    def __init__(self, filepaths, transform=None):
        super(CocoLoader, self).__init__(
            filepaths[0],
            filepaths[1],
            transform=transform,
        )

    def get_api(self):
        return self.coco

    def decode_ann(self, ann):
        return self.coco.annToMask(ann)

    def get_amount(self, amount, offset=0):
        assert offset >= 0
        assert amount > 0
        return [self[offset + i] for i in range(amount)]

    def __getitem__(self, index):
        item = super(CocoLoader, self).__getitem__(index)
        img = item[0]
        annotations = item[1]

        boxes = []
        masks = []
        cats = []
        for ann in annotations:
            box = utils.box_coco_to_sam(ann["bbox"])
            boxes.append(box)
            mask = self.decode_ann(ann)
            masks.append(mask)
            cats.append(ann["category_id"])

        boxes = np.array(boxes)
        masks = np.array(masks)

        return {
            "image": pil_to_tensor(img),
            "annotations": (
                {
                    "boxes": torch.Tensor(boxes),
                    "masks": torch.Tensor(masks),
                    "categories": torch.Tensor(cats),
                }
            ),
        }

    def translate_catIDs(self, catIDs):
        return [self.coco.cats[int(catID)]["name"] for catID in catIDs]


def test_coco_loading():
    print("\nTesting coco loading")

    paths = get_coco_split("train", "2017")
    coco = CocoLoader(paths, transform=None)
    item = coco[0]
    assert item["image"].shape[1:] == item["annotations"]["masks"].shape[1:]
    assert item["image"].shape[0] == 3
    assert len(coco.get_amount(10)) == 10
    print(coco.translate_catIDs([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))


if __name__ == "__main__":
    test_coco_loading()
