import torch
import os
import numpy as np
import torchvision.datasets
from torchvision.datasets import CocoDetection
from torchvision.transforms.functional import pil_to_tensor
import pycocotools
from torch.utils.data import DataLoader

import utils  # otherwise does not work from main repo


def box_coco_to_sam(coco_box):
    """
    Convert coco box to sam box
    from x0,y0,w,h to x0,y0,x1,y1
    """
    return (
        round(coco_box[0]),
        round(coco_box[1]),
        round((coco_box[0] + coco_box[2])),
        round((coco_box[1] + coco_box[3])),
    )


def get_coco_split(split: str = "train", year: str = "2017", root=None):
    """
    Returns the paths of the COCO dataset split for a given year.
    Args:
        split (str, optional): The split of the dataset. Defaults to "train".
        year (str, optional): The year of the dataset. Defaults to "2017".
        root is the root directory of datasets
    Returns:
        tuple: A tuple containing the image path and annotation path.
    """

    assert split in ["train", "val"]
    assert year in ["2014", "2017"]
    if root is None:
        root = "/mnt/vrg2/imdec/datasets/COCO"  # default root

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
        self.new_class_ids = [
            index for index, previsous in enumerate(self.coco.cats.keys())
        ]
        self.old_to_new_cat_translation_table = {
            old_id: new_id
            for new_id, old_id in zip(self.new_class_ids, self.coco.cats.keys())
        }
        self.new_to_old_cat_translation_table = {
            new_id: old_id
            for new_id, old_id in zip(self.new_class_ids, self.coco.cats.keys())
        }

    def get_cat_keys(self):  # new categories, without gaps
        return self.new_class_ids

    def translate_to_new_catID(self, catID):  # from old category to its new category
        return self.old_to_new_cat_translation_table[catID]

    def translate_to_old_catID(self, catID):  # from new category to its old category
        return self.new_to_old_cat_translation_table[catID]

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
            box = box_coco_to_sam(ann["bbox"])
            boxes.append(box)
            mask = self.decode_ann(ann)
            masks.append(np.uint8(mask))
            cats.append(self.translate_to_new_catID(ann["category_id"]))
            # to normalize categories. base cats have a lot of gaps

        boxes = np.array(boxes)
        masks = np.array(masks)

        return {
            "image": np.asarray(img),
            "annotations": (
                {
                    "boxes": torch.Tensor(boxes).type(
                        torch.int16
                    ),  # cant be uint8!! weird error with conversion
                    "masks": torch.Tensor(masks).type(torch.uint8),
                    "categories": torch.Tensor(cats),
                }
            ),
        }

    def translate_catIDs(self, catIDs):
        return [
            self.coco.cats[self.translate_to_old_catID(int(catID))]["name"]
            for catID in catIDs
            # again need to map to original catIDs and return the names
        ]

    def instantiate_loader(self, batch_size=4, num_workers=4):

        def collate_fn(batch):
            images = [item["image"] for item in batch]
            annotations = [item["annotations"] for item in batch]
            return images, annotations

        data_loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,  # dont mix pls. we then save the positions
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        return data_loader

    def get_classes(self):
        return [self.coco.cats[int(catID)]["name"] for catID in self.coco.cats.keys()]


def test_coco_loading():
    print("\nTesting coco loading")

    paths = get_coco_split(split="val", year="2017")
    coco = CocoLoader(paths, transform=None)

    item = coco[0]
    assert item["image"].shape[:2] == item["annotations"]["masks"].shape[1:]
    assert item["image"].shape[2] == 3
    assert len(coco.get_amount(10)) == 10

    print(coco.translate_catIDs([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    print(coco.get_classes())
    print(coco.get_cat_keys())
    print("Coco loading test passed!")


if __name__ == "__main__":
    test_coco_loading()
    # if problem, run as python -m datasets.dataset_loading
