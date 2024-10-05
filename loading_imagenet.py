import numpy as np
import json
import os
import pprint
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd
import datasets

from torchvision.datasets import ImageNet


class ImagenetResults:

    def __init__(self, path):
        self.path = path.strip()

        if not os.path.exists(self.path):
            print("Directory does not exist.")
            return

        self.dataset = self.load_same_dataset()

        # prepare dataset from config, which is loaded from the folder

    def load_same_dataset(self):
        """
        loads the same dataset which was used for inference, based on config
        """
        cfg = self.load_config()["dataset"]
        split = cfg["split"]
        root = cfg["root"]  # like "/mnt/vrg2/imdec/datasets/ImageNet/imagenet_pytorch"

        dataset = ImageNet(root=root, split=split)
        return dataset

    def load_config(self, print_conf=False):
        with open(os.path.join(self.path, "config.json"), "r") as f:
            config = json.load(f)
        if print_conf:
            pprint.pprint(config)
        return config

    def load_results_per_image(self, idx, folder_name="masks_detections"):
        """
        idx : id of image to ge loaded
        folder_name : folder where the detections are saved inside the main folder (where config and results are saved)
            - if you dont have different structure, then dont provide any, default is fine
        """
        path = os.path.join(self.path, folder_name, f"detections_{idx}.npz")

        if not os.path.exists(path):
            print(f"Folder {path} does not exist.")
            return None

        # load boxes and masks, return in a dict
        nzp = np.load(path)
        image = self.dataset[idx][0]
        image = np.asarray(image)
        return {
            "image": image,
            "boxes": nzp["boxes"],
            "masks": nzp["masks"],
            "labels": nzp["labels"],
            "areas": nzp["areas"],
        }


if __name__ == "__main__":
    result_loader = ImagenetResults(path="/datagrid/fix_in/detection/imagenet_val")
    pprint.pprint(result_loader.load_config())
    for i in range(10):
        image_dict = result_loader.load_results_per_image(i)
        print(image_dict.keys())

        print(f"shape of image {i}: ", image_dict["image"].shape)
        print(
            "labels: ", image_dict["labels"]
        )  # -1 if none good class predicted from detector
        print("len of masks: ", len(image_dict["masks"]))
        print("len of boxes: ", len(image_dict["boxes"]))
