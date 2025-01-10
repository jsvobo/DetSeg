import torch


bbox_area_ranges = {
    "S": (float(0**2), float(32**2)),
    "M": (float(32**2), float(96**2)),
    "L": (float(96**2), float(1e5**2)),
}
# as in https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/detection/_mean_ap.py


def find_range(area):
    """
    Find the range of the area of a bounding box
    """
    for key, value in bbox_area_ranges.items():
        if value[0] <= area < value[1]:
            return key
    return "larger"


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


def print_for_task(d):
    avg_iou = float(d["avg iou"])
    print(f" TP: {d['TP']}, FP: {d['FP']}, FN: {d['FN']}")
    print(f" Precision:         {round(d['Precision'],3)}")
    print(f" Recall:            {round(d['Recall'],3)}")
    print(f" F1-Measure:        {round(d['F1'],3)}")
    print(f" Average IoU:       {round(avg_iou,3) }")
    subdict_ca = d["mAP without classes"]
    print("     mAP: ", subdict_ca["map"])
    print("     mAR - small: ", subdict_ca["mar_small"])
    print("     mAR - medium: ", subdict_ca["mar_medium"])
    print("     mAR - large: ", subdict_ca["mar_large"])
