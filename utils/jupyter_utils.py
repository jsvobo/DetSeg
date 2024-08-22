import matplotlib.pyplot as plt
import torch
import numpy as np
import utils
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


global_titlesize = 15  # dont touch? not important


def show_points(coords, labels, ax, marker_size=300):
    """
    Interpreted from https://github.com/facebookresearch/segment-anything
    """
    pos_points = coords[np.where(labels == 1)]
    neg_points = coords[np.where(labels == 0)]

    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )

    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def plot_box(box, ax, color="red", linewidth=2):
    """
    bounding boxes in format x0,y0,x1,y1 (main diagonal points)
    Plots the boc on the ax from plt
    """
    ax.plot(
        [box[0], box[2], box[2], box[0], box[0]],
        [box[1], box[1], box[3], box[3], box[1]],
        color=color,
        linewidth=linewidth,
    )


def grid_masks_boxes(
    image,
    masks,
    boxes,
    titles=None,
    scale=8,
    linewidth=3,
    points=None,
    point_labels=None,
):
    num_imgs = len(masks) + 1
    fig, axes = plt.subplots(1, num_imgs, figsize=(num_imgs * scale, scale))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i == 0:  # straight up image
            ax.imshow(image)
            ax.axis("off")
            ax.set_title("Base image", fontsize=global_titlesize)
            continue

        index = i - 1
        ax.axis("off")
        ax.imshow(image)
        ax.set_title(titles[index], fontsize=global_titlesize)

        if points is not None:
            assert len(points) == len(point_labels)
            if points[index] is not None:
                utils.show_points(
                    coords=points[index], labels=point_labels[index], ax=ax
                )

        utils.plot_box(boxes[index], ax, linewidth=linewidth)
        ax.imshow(1 * masks[index], cmap="jet", alpha=0.5)
        # plot this last so the bbox is on top of the mask and not in the edges

    plt.show()


def print_masks_boxes(
    image,
    masks,
    boxes,
    linewidth=3,
    scale=8,
    opacity=0.8,
    mask_background=False,
    colormap_name="viridis",
):
    """
    Bounding boxes in format x0,y0,x1,y1 (main diagonal points)
    prints all masks and boxes on the image
    if None is provided, no boxes are printed, same for masks
    """
    plt.figure(figsize=(scale, scale))
    plt.imshow(image)  # first image
    plt.axis("off")

    has_masks = (masks is not None) and (len(masks) > 0)
    has_boxes = (boxes is not None) and (len(boxes) > 0)

    if has_masks and has_boxes:  # I have both
        assert len(masks) == len(boxes)

    if has_masks:
        cmap = colormaps[colormap_name]
        alpha = np.ones_like(image)[:, :, 0] * opacity
        mask_sum = np.zeros_like(image)[:, :, 0]

        for i, mask in enumerate(masks):
            mask_sum = np.maximum(mask_sum, mask * (i + 1))  # layer masks
        if not mask_background:
            alpha[np.where(mask_sum == 0)] = 0
        else:
            alpha[np.where(mask_sum == 0)] = opacity / 2
        plt.imshow(mask_sum, cmap=cmap, alpha=alpha)

    if has_boxes:
        num_boxes = len(boxes)
        for i, box in enumerate(boxes):  # all masks
            utils.plot_box(box, plt.gca(), linewidth=linewidth)

    plt.show()


# color list, patches and colormap definitions for mask visualisation, function show_differences()
colors1 = ["blue", "red"]
patches1 = [mpatches.Patch(color=c) for c in colors1]
cmap1 = ListedColormap(colors1)
labels1 = ["Background", "Mask"]

colors2 = ["blue", "red", "yellow", "green"]
patches2 = [mpatches.Patch(color=c) for c in colors2]
cmap2 = ListedColormap(colors2)
labels2 = ["Background", "Only inferred", "Only GT", "Intersection"]


def show_differences(
    dict_bad_mask,
    scale=8,
    linewidth=2,
    gt_class=None,
    title_size=22,
    opacity=0.5,
    segmentation_model="SAM-1",
):
    # load from dict
    image = dict_bad_mask["image"]
    box = dict_bad_mask["box"]
    inferred_mask = dict_bad_mask["inferred_mask"]
    gt_mask = dict_bad_mask["gt_mask"]

    # listing for every picture, first is just the image
    cmaps = [cmap1, cmap1, cmap2]
    patches = [patches1, patches1, patches2]
    labels = [labels1, labels1, labels2]
    ncols = [1, 1, 2]
    titles = ["GT", segmentation_model, "Overlapping masks"]

    if gt_class is not None:
        # change titles to include the gt class names
        titles[0] = "GT: " + gt_class

    # Base image
    fig, axes = plt.subplots(1, 4, figsize=(4 * scale, scale))
    ax = axes[0]
    utils.plot_box(box, ax, color="red", linewidth=linewidth)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title("Base image", fontsize=title_size)

    # Image with masks (GT,Inferred, Overlapping)
    both_masks = np.int64(inferred_mask) + 2 * np.int64(gt_mask)  # adds layers
    masks_to_show = [gt_mask, inferred_mask, both_masks]
    for i in range(3):
        ax = axes[i + 1]
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(titles[i], fontsize=title_size)
        utils.plot_box(box, ax)
        ax.imshow(masks_to_show[i], cmap=cmaps[i], alpha=opacity)
        ax.legend(patches[i], labels[i], ncols=ncols[i], fontsize="x-large")

    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.show()
