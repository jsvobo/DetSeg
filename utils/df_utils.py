import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import utils

import pprint
import datasets

show = 10  # how many rows to show in the dataframe


def _aggregate_by_sizes(frame, to_print=True):
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


def aggregate_missed_by_classes(frame, dataset, to_print=True):
    filtered = frame[frame["iou"] == 0]

    class_indices = dataset.get_cat_keys()
    category_names = dataset.get_classes()

    frequencies = [len(filtered[filtered["gt_class"] == idx]) for idx in class_indices]
    totals = [len(frame[frame["gt_class"] == idx]) for idx in class_indices]
    percentage_missed = [f / t if t != 0 else 0 for f, t in zip(frequencies, totals)]

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


def do_overview(loader):
    full_metrics = loader.load_metrics()
    config = loader.load_config()
    frame_boxes = loader.load_matched_boxes()
    frame_masks = loader.load_matched_masks()

    # main info
    name = config["class_list"]["name"]
    print("prompt list: ", name)

    # print selected result metrics
    print("Detection:")
    det = full_metrics["detection"]
    utils.print_for_task(d=det)

    print("Segmentation:")
    seg = full_metrics["segmentation"]
    utils.print_for_task(d=seg)

    # show histograms for boxes and masks without non-matches
    utils.plot_box_histogram(
        frame_boxes[frame_boxes["iou"] != 0], name, object_type="boxes"
    )
    utils.plot_box_histogram(
        frame_masks[frame_masks["iou"] != 0], name, object_type="masks"
    )


def load_missed_aggregate_per_class(loader, dataset):
    frame_boxes = loader.load_matched_boxes()
    frame_masks = loader.load_matched_masks()

    # aggregate boxes and masks
    boxes = aggregate_missed_by_classes(frame_boxes, dataset, to_print=False)
    masks = aggregate_missed_by_classes(frame_masks, dataset, to_print=False)

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


def prompt_sets_comparison(paths, names, dataset):
    absolute = {}
    percentage = {}

    # unwrap paths and names, load frames and concatenate into a df
    for path, name in zip(paths, names):  # dont calculate area here! no area!
        b = pd.read_pickle(os.path.join(path, "boxes_df.pkl"))  # load directly
        b = aggregate_missed_by_classes(b, dataset, to_print=False)
        absolute[name] = b["missed"]
        percentage[name] = b["part_missed"]

    # implant categories and totals into the df
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
