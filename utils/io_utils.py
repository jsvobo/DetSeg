import numpy as np
import json
import os
import time
import pprint
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd


def save_results(results, cfg):
    """
    Save the results of the pipeline to a file
    """
    metrics = results["metrics"]
    boxes_df = results["boxes_df"]
    masks_df = results["masks_df"]

    # compose folder name
    dir_path = cfg.save_path
    det_name = cfg.detector.class_name
    seg_name = cfg.segmentation.class_name + "_" + cfg.segmentation.sam_model
    num_data = int(cfg.batch_size * cfg.max_batch)
    prompts = cfg.class_list.name
    split = cfg.dataset.split
    dataset_name = cfg.dataset.name
    folder_name = f"{det_name}_{seg_name}_{prompts}_{dataset_name}_{split}_{num_data}_{time.strftime('%m_%d')}"
    path = dir_path + folder_name

    # save result dicts and config dict
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "results.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(cfg), f, indent=4)

    # save individual IoU results as pickle
    boxes_df_pickle = boxes_df.to_pickle(os.path.join(path, "boxes_df.pkl"))
    masks_df_pickle = masks_df.to_pickle(os.path.join(path, "masks_df.pkl"))

    print(f"Results saved to {path}")


def load_results(dir_path, print_conf=False):
    """
    Load the saved results from a folder
    """
    if not os.path.exists(dir_path):
        print("Directory does not exist.")
        return

    with open(os.path.join(dir_path, "results.json"), "r") as f:
        results = json.load(f)
    with open(os.path.join(dir_path, "config.json"), "r") as f:
        config = json.load(f)
        if print_conf:
            pprint.pprint(config)

    # load individual IoU results from pickle
    array_boxes = pd.read_pickle(os.path.join(dir_path, "boxes_df.pkl"))
    array_masks = pd.read_pickle(os.path.join(dir_path, "masks_df.pkl"))

    return results, config, array_boxes, array_masks


def convert_tensors_to_save(d):
    """
    Recursively convert dictionary with tensors to dictionary with lists at the leaves.
    This is done for saving purposes, as torch.Tensor cannot be saved to disk
    """
    if isinstance(d, dict):
        # Recursively apply the function for nested dictionaries
        return {k: convert_tensors_to_save(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        # Convert the torch.Tensor to a numpy array
        return d.cpu().tolist()
    else:
        # Return the value as is if it's neither a dict nor a torch.Tensor
        return d
