import matplotlib.pyplot as plt
import torch
import numpy as np
import utils
from matplotlib import colormaps


global_titlesize = 15  # dont touch? not important


def to_plt_order(image: torch.Tensor):
    """
    Reshape a torch tensor (chw) to order hwc for matplotlib plotting.
    Args:
        torch.Tensor: Input image tensor in chw format on CPU.
    Returns:
        image (torch.Tensor): Reshaped image tensor in hwc format.
    """
    return image.permute(1, 2, 0).cpu()


def to_torch_order(image: torch.Tensor):
    """
    Reshape hwc array to chw for torch calculations.
    Args:
        image (torch.Tensor): Input image tensor in hwc format.
    Returns:
        torch.Tensor: Reshaped image tensor in chw format on CPU.
    """
    return image.permute(2, 0, 1).cpu()


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


def grid_masks_boxes(image, masks, boxes, titles=None, scale=8, linewidth=3):
    num_imgs = len(masks) + 1
    image = to_plt_order(image)  # reorders image to HWC from torch CHW and moves to cpu
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
        ax.imshow(masks[index], cmap="jet", alpha=0.5)
        utils.plot_box(boxes[index], ax, linewidth=linewidth)

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
    image = to_plt_order(image)
    plt.figure(figsize=(scale, scale))
    plt.imshow(image)  # first image
    plt.axis("off")

    has_masks = (masks is not None) and masks != []
    has_boxes = (boxes is not None) and boxes != []

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
