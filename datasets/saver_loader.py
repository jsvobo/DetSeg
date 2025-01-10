import numpy as np
import json
import os
import time
import pprint
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd
import utils
import datasets

import torchvision.transforms as T
from PIL import Image
from torchvision.io import write_png


class ResultSaver:

    def __init__(self, cfg):
        self.cfg = cfg

        if not self.cfg.save_results:  # dont save at all
            return

        # compose folder name from config
        dir_path = self.cfg.save_path
        det_name = self.cfg.detector.class_name
        seg_name = (
            self.cfg.segmentation.class_name + "_" + self.cfg.segmentation.sam_model
        )
        num_data = (
            int(self.cfg.batch_size * self.cfg.max_batch)
            if self.cfg.max_batch is not None
            else "full"
        )
        prompts = self.cfg.class_list.name
        split = self.cfg.dataset.split
        dataset_name = self.cfg.dataset.name
        folder_name = f"{det_name}_{seg_name}_{prompts}_{dataset_name}_{split}_{num_data}_{time.strftime('%m_%d')}"

        # important paths
        self.path = dir_path + folder_name
        self.masks_detections_folder = os.path.join(self.path, "masks_detections")

        # make directories if not exists
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.masks_detections_folder, exist_ok=True)

    def save_per_image(self, boxes, masks, labels, image_id):
        subfolder_name = f"detections_{image_id}"
        folder_name = os.path.join(self.masks_detections_folder, subfolder_name)

        areas = self.calculate_area(boxes)

        np.savez_compressed(  # adds .nzp extension
            folder_name, areas=areas, boxes=boxes, masks=masks, labels=labels
        )

    def calculate_area(self, boxes):
        area_boxes = [utils.area(box) for box in boxes]
        area_type = [utils.find_range(area) for area in area_boxes]
        return area_type

    def save_results(self, results):
        path = self.path

        # separate all results back to individual dataframes
        metrics = results["metrics"]
        boxes_df = results["boxes_df"]
        masks_df = results["masks_df"]
        image_level_df = results["image_level_df"]

        # save result and config dicts
        with open(os.path.join(path, "results.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(self.cfg), f, indent=4)

        # calculate area type for every box.
        gt_boxes = boxes_df["gt"]
        area_type = self.calculate_area(gt_boxes)
        boxes_df["area"] = area_type
        masks_df["area"] = area_type

        # save the number of detections per image (not what detections yet)
        image_level_df[["image_id", "num_detections"]].to_pickle(
            os.path.join(path, "image_df.pkl")
        )

        # save individual IoU results as pickle
        boxes_df.to_pickle(os.path.join(path, "boxes_df.pkl"))
        masks_df.to_pickle(os.path.join(path, "masks_df.pkl"))

        print(
            f"\nResults saved to {path} \n with files:\
image_df.pkl, boxes_df.pkl, masks_df.pkl,\
config.json, results.json, and folder masks_detections.\
\nThis folder has a file for *every image* named detections_<image_id>"
        )


class ResultLoader:

    def __init__(self, path):
        self.path = path.strip()

        if not os.path.exists(self.path):
            print("Directory does not exist.")
            return

        # prepare dataset from config, which is loaded from the folder

    def load_same_dataset(self):
        """
        loads the same dataset which was used for inference, based on config
        """
        cfg = self.load_config()["dataset"]
        split = cfg["split"]
        year = cfg["year"]
        root = cfg["root"]
        dataset_class = cfg["class_name"]
        get_path = cfg["split_fn"]

        if "transforms" in cfg.keys() and cfg["transforms"] != "None":
            transforms = cfg["transforms"]
            # get the specific transforms from datasets module
            transforms = getattr(datasets, transforms)()
        else:
            transforms = None

        dataset_path = getattr(datasets, get_path)(split=split, year=year, root=root)
        return getattr(datasets, dataset_class)(dataset_path, transform=transforms)

    def load_matched_boxes(self):
        boxes_df = pd.read_pickle(os.path.join(self.path, "boxes_df.pkl"))
        return boxes_df

    def load_matched_masks(self):
        masks_df = pd.read_pickle(os.path.join(self.path, "masks_df.pkl"))
        return masks_df

    def load_image_level(self):
        image_level = pd.read_pickle(os.path.join(self.path, "image_df.pkl"))
        return image_level

    def load_config(self, print_conf=False):
        with open(os.path.join(self.path, "config.json"), "r") as f:
            config = json.load(f)
        if print_conf:
            pprint.pprint(config)
        return config

    def load_metrics(self):
        with open(os.path.join(self.path, "results.json"), "r") as f:
            results = json.load(f)
        return results

    def load_all(self):
        metrics = self.load_metrics()
        config = self.load_config()
        array_boxes = self.load_matched_boxes()
        array_masks = self.load_matched_masks()
        array_image_level = self.load_image_level()

        return {
            "metrics": metrics,
            "config": config,
            "boxes_df": array_boxes,
            "masks_df": array_masks,
            "image_level_df": array_image_level,
        }

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
        return {
            "boxes": nzp["boxes"],
            "masks": nzp["masks"],
            "labels": nzp["labels"],
            "areas": nzp["areas"],
        }
