import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import utils

import pprint
from datasets.dataset_loading import CocoLoader, get_coco_split


show = 10


def plot_box_histogram(df, name, object_type):
    iou_array = df["iou"]
    lspace = np.linspace(0.0, 1.0, 100)

    # Calculate the number of objects in each bin
    hist, bins, _ = plt.hist(iou_array, bins=lspace)

    # Find the bin index where the mean value of IoU_array falls
    mean_threshold = np.mean(iou_array)
    median_threshold = np.median(iou_array)
    plt.axvline(x=mean_threshold, color="red", linestyle="--")
    plt.axvline(x=median_threshold, color="black", linestyle="--")
    # Add labels and title
    plt.xlabel("IoU")
    plt.ylabel("Frequency")

    size = len(iou_array)
    plt.title(f"{name}, per GT instance IoU histogram: {object_type}")  # {size} samples

    mean = "{:.2f}".format(mean_threshold)
    median = "{:.2f}".format(median_threshold)
    plt.legend([f"Mean - {mean}", f"Median - {median}"])
    # Show the plot
    plt.show()


def aggregate_by_sizes(frame, to_print=True):
    # aggregate misses by area

    # filter out GT with no matches
    filtered = frame[frame["iou"] == 0]
    aggr = filtered.groupby("area").size()  # .size()

    category_names = ["L", "M", "S"]
    frequencies = [aggr.get(i) for i in aggr.index]

    # aggregate all the boxes, not just bad ones, count all class occurences
    aggr_whole = frame.groupby("area").size()
    totals = [aggr_whole.get(i) for i in aggr.index]
    percentage_missed = [f / t for f, t in zip(frequencies, totals)]

    df = pd.DataFrame(
        {
            "category": category_names,
            "missed": frequencies,
            "total": totals,
            "part_missed": percentage_missed,
        }
    )
    if to_print:
        df.sort_values(by="missed", ascending=False, inplace=True)
        display(df[:show])

        df.sort_values(by="part_missed", ascending=False, inplace=True)
        display(df[:show])

    return df.sort_index()


def aggregate_missed_by_classes(frame, coco_dataset, to_print=True):
    filtered = frame[frame["iou"] == 0]

    class_indices = coco_dataset.get_cat_keys()
    category_names = coco_dataset.get_classes()

    frequencies = [len(filtered[filtered["gt_class"] == idx]) for idx in class_indices]
    totals = [len(frame[frame["gt_class"] == idx]) for idx in class_indices]
    percentage_missed = [f / t for f, t in zip(frequencies, totals)]

    df = pd.DataFrame(
        {
            "category": category_names,
            "missed": frequencies,
            "total": totals,
            "part_missed": percentage_missed,
        }
    )

    if to_print:
        df.sort_values(by="missed", ascending=False, inplace=True)
        display(df[:show])

        df.sort_values(by="part_missed", ascending=False, inplace=True)
        display(df[:show])

    return df.sort_index()


def print_subresults(results):
    print("     mAP: ", results["map"])
    print("     mAR - small: ", results["mar_small"])
    print("     mAR - medium: ", results["mar_medium"])
    print("     mAR - large: ", results["mar_large"])


def do_overview(path, name="Coco classes", coco_dataset=None):
    # load data
    loaded_dict = utils.load_results(path, print_conf=False)

    # parse back
    config = loaded_dict["config"]
    frame_boxes = loaded_dict["boxes_df"]
    frame_masks = loaded_dict["masks_df"]
    results = loaded_dict["results"]
    frame_images = loaded_dict["image_level_df"]

    # missed parts
    miss_box = len(frame_boxes[frame_boxes["iou"] == 0])
    miss_mask = len(frame_masks[frame_masks["iou"] == 0])
    total_GTs = len(frame_boxes)

    # main info
    print("prompt list: ", config["class_list"]["name"])

    # total detections?
    total_detections = frame_images["num_detections"].sum()
    print("\nTotal detections: ", total_detections)
    print("Total GTs: ", total_GTs)
    print("Detection FP: ", total_detections - total_GTs + miss_box)
    print("Segementation FP: ", total_detections - total_GTs + miss_mask)

    # precisiona nd recall: baxes and masks
    TP_boxes = total_GTs - miss_box
    precision_box = TP_boxes / total_detections
    recall_box = TP_boxes / total_GTs

    TP_masks = total_GTs - miss_mask
    precision_mask = TP_masks / total_detections
    recall_mask = TP_masks / total_GTs

    print("\nDetection precision: ", precision_box)
    print("Detection recall: ", recall_box)
    print("Segmentation precision: ", precision_mask)
    print("Segmentation recall: ", recall_mask)

    frac_miss_box = miss_box / total_GTs
    frac_miss_mask = miss_mask / total_GTs

    print("\nfraction of GTs missed, det and seg:")
    print(frac_miss_box)
    print(frac_miss_mask)

    # print selected result metrics
    print("\ndetection average IoU: ", results["average det IoU"])
    print_subresults(results["classless mAP - detection"])
    print("segmentation average IoU: ", results["average seg IoU"])
    print_subresults(results["classless mAP - detection"])

    # show histograms for boxes and masks without non-matches
    plot_box_histogram(frame_boxes[frame_boxes["iou"] != 0], name, object_type="boxes")
    plot_box_histogram(frame_masks[frame_masks["iou"] != 0], name, object_type="masks")


def load_missed_aggregate_per_class(path, coco_dataset):
    results, config, frame_boxes, frame_masks = utils.load_results(
        path, print_conf=False
    )

    # aggregate boxes and masks
    boxes = aggregate_missed_by_classes(frame_boxes, coco_dataset, to_print=False)
    masks = aggregate_missed_by_classes(frame_masks, coco_dataset, to_print=False)

    return {
        "boxes_per_class": boxes,
        "masks_per_class": masks,
        "frame_boxes": frame_boxes[frame_boxes["iou"] == 0],
        "frame_masks": frame_masks[frame_masks["iou"] == 0],
    }


def squish_df(df):
    axis_names = df.keys()
    df = pd.concat(df.values(), axis=1)
    df.columns = axis_names
    return df


def comparison(paths, names, coco_dataset):
    absolute = {}
    percentage = {}

    # unwrap paths and names, load frames and concatenate into a df
    for path, name in zip(paths, names):  # dont calculate area here! no area!
        b = pd.read_pickle(os.path.join(path, "boxes_df.pkl"))  # load directly
        b = aggregate_missed_by_classes(b, coco_dataset, to_print=False)
        absolute[name] = b["missed"]
        percentage[name] = b["part_missed"]

    # implant categories and totals
    cats = b["category"]
    totals = b["total"]
    absolute["Category"] = cats
    percentage["Category"] = cats
    absolute["Totals"] = totals
    percentage["Totals"] = totals

    # squish into into one df each
    absolute = squish_df(absolute)
    percentage = squish_df(percentage)

    return absolute, percentage
