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


def aggregate_by_coco_classes(frame, coco_dataset, to_print=True):
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
    # load data and maybe
    results, config, frame_boxes, frame_masks = utils.load_results(
        path, print_conf=False
    )
    print("prompt list: ", config["class_list"]["name"])
    print("# missed GTs, det and seg:")
    print(len(frame_boxes[frame_boxes["iou"] == 0]))
    print(len(frame_masks[frame_masks["iou"] == 0]))

    # print selected results
    print("\ndetection mean IoU: ", results["mean det IoU"])
    print_subresults(results["classless mAP - detection"])
    print("segmentation mean IoU: ", results["mean seg IoU"])
    print_subresults(results["classless mAP - detection"])

    # show histograms for boxes and masks
    plot_box_histogram(frame_boxes, name, object_type="boxes")
    plot_box_histogram(frame_masks, name, object_type="masks")

    # aggregate results using classes and sizes
    classes_boxes = aggregate_by_coco_classes(frame_boxes, coco_dataset)
    sizes_boxes = aggregate_by_sizes(frame_boxes)

    # reorder by total per class
    classes_boxes.groupby(by="total")
    return classes_boxes


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
        b = aggregate_by_coco_classes(b, coco_dataset, to_print=False)
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
