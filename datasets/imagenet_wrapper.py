import torch
import os
import numpy as np

from torchvision.datasets import ImageNet
from torchvision.transforms.functional import pil_to_tensor
import pycocotools
from torch.utils.data import DataLoader


def get_imagenet_split(split: str = "train", year: str = "2017", root=None):
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

    if root is None:
        print(
            "No root provided, using default: /mnt/vrg2/imdec/datasets/ImageNet/imagenet_pytorch"
        )
        root = "/mnt/vrg2/imdec/datasets/ImageNet/imagenet_pytorch"  # default root of none is provided
    return (root, split)


class ImagenetLoader(ImageNet):

    def __init__(self, filepaths, transform=None):
        super().__init__(
            root=filepaths[0],
            split=filepaths[1],
            transform=transform,
        )

    def get_cat_keys(self):
        # all class numbers
        return self.wnids

    def get_classes(self):
        # return all category names
        return self.classes

    # def CatID_old_to_new(self, catID):
    #     # irrelevant here, justo for coco, need this to be backwards compatible?
    #     return catID

    # def CatID_new_to_old(self, catID):
    #     # irrelevant here, justo for coco, need this to be backwards compatible?
    #     return catID

    def class_name_to_new_ID(self, name):
        class_dict = self.class_to_idx
        if name in class_dick.keys():
            return self.class_to_idx[name]
        return -1

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

        return {
            "image": np.asarray(img),
            "annotations": (
                {  # boxes cant be uint8!! weird error with conversion
                    "boxes": torch.Tensor([]).type(torch.int32),
                    "masks": torch.Tensor([]).type(torch.bool),
                    "categories": torch.Tensor([]).type(torch.int16),
                }  # imagenet has no masks or boxes! metrics will be useless then, but maybe we can save the detections
            ),
            "index": index,
        }

    def catIDs_to_names(self, catIDs):
        return [self.classes[int(catID)] for catID in catIDs]

    def create_dataloader(self, batch_size=4, num_workers=4):

        def collate_fn(batch):
            images = [item["image"] for item in batch]
            annotations = [item["annotations"] for item in batch]
            idcs = [item["index"] for item in batch]
            return images, annotations, idcs

        data_loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        return data_loader


def test_imagenet_loading():
    print("\nTesting coco loading")

    paths = get_imagenet_split(split="val")
    dataset = ImagenetLoader(filepaths=paths, transform=None)

    item = dataset[1]
    assert item["image"].shape[2] == 3
    assert len(dataset.get_amount(10)) == 10

    print(dataset.catIDs_to_names([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    print(dataset.get_classes())
    print(dataset.get_cat_keys())
    print("Imagenet loading test passed!")


if __name__ == "__main__":
    test_imagenet_loading()
    # if problem, run as python -m datasets.dataset_loading
