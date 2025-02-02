import torch
import os
import numpy as np
import torchvision.datasets
from torchvision.datasets import CocoDetection
from torchvision.transforms.functional import pil_to_tensor
import pycocotools
from torch.utils.data import DataLoader
import utils


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
        print("No root provided, using default: /mnt/vrg2/imdec/datasets/COCO")
        root = "/mnt/vrg2/imdec/datasets/COCO"  # default root of none is provided

    annotation_path = os.path.join(root, "annotations")
    type_ann = "instances"  # instances, captions, person_keypoints

    ann_path = os.path.join(annotation_path, type_ann + "_" + split + year + ".json")
    image_path = os.path.join(root, split + year)
    return (image_path, ann_path)


class CocoLoader(CocoDetection):

    def __init__(self, filepaths, transform=None):
        super().__init__(
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

    def CatID_old_to_new(self, catID):  # from old category to its new category
        return self.old_to_new_cat_translation_table[catID]

    def CatID_new_to_old(self, catID):  # from new category to its old category
        return self.new_to_old_cat_translation_table[catID]

    def class_name_to_new_ID(self, name):
        list_names = self.get_classes()
        if name in list_names:
            return list_names.index(name)
        else:
            return -1

    def get_api(self):
        return self.coco

    def coco_ann_to_binary_mask(self, ann):
        """
        decodes mask from coco annotation, uses pycocotools api class to do this
        """
        return self.coco.annToMask(ann)

    def get_amount(self, amount, offset=0):
        """
        Returns a list of amount items from the dataset, starting from offset
        Example: In notebook, I want to explore 10 images from the dataset,
             this fn loads them all at once
        """
        assert offset >= 0
        assert amount > 0
        return [self[offset + i] for i in range(amount)]

    def __getitem__(self, index):
        img, annotations = super().__getitem__(index)

        boxes, masks, cats = [], [], []
        for ann in annotations:
            box = utils.box_coco_to_sam(ann["bbox"])
            boxes.append(box)
            mask = self.coco_ann_to_binary_mask(ann)
            masks.append(np.uint8(mask))
            cats.append(self.CatID_old_to_new(ann["category_id"]))
            # to normalize categories. base cats have a lot of gaps

        boxes = np.array(boxes)
        masks = np.array(masks)

        return {
            "image": np.asarray(img),
            "annotations": (
                {  # boxes cant be uint8!! weird error with conversion
                    "boxes": torch.Tensor(boxes).type(torch.int32),
                    "masks": torch.Tensor(masks).type(torch.bool),
                    "categories": torch.Tensor(cats).type(torch.int16),
                }
            ),
            "index": index,
        }

    def catIDs_to_names(self, catIDs):
        return [
            self.coco.cats[self.CatID_new_to_old(int(catID))]["name"]
            for catID in catIDs
            # again need to map to original catIDs and return the names
        ]

    def create_dataloader(self, batch_size=4, num_workers=4):

        def collate_fn(batch):
            images = [item["image"] for item in batch]
            annotations = [item["annotations"] for item in batch]
            idcs = [item["index"] for item in batch]
            return images, annotations, idcs

        data_loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,  # dont mix pls. we then save the positions TODO: return indices?
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        return data_loader

    def get_classes(self):
        # return all category names
        return [self.coco.cats[int(catID)]["name"] for catID in self.coco.cats.keys()]


def test_coco_loading():
    print("\nTesting coco loading")

    paths = get_coco_split(split="val", year="2017")
    coco = CocoLoader(paths, transform=None)

    item = coco[0]
    assert item["image"].shape[:2] == item["annotations"]["masks"].shape[1:]
    assert item["image"].shape[2] == 3
    assert len(coco.get_amount(10)) == 10

    print(coco.catIDs_to_names([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    print(coco.get_classes())
    print(coco.get_cat_keys())
    print("Coco loading test passed!")


if __name__ == "__main__":
    test_coco_loading()
    # if problem, run as python -m datasets.coco_wrapper
