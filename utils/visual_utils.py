import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from PIL import Image
import utils


# color list, patches and colormap definitions for mask visualisation (sam.ipynb)
colors_together = ["blue", "red", "yellow", "green"]
patches_together = [
    mpatches.Patch(color=colors_together[i]) for i in range(len(colors_together))
]
cmap_together = ListedColormap(colors_together)

colors_separate = ["blue", "red"]
patches_separate = [
    mpatches.Patch(color=colors_separate[i]) for i in range(len(colors_separate))
]
cmap_single = ListedColormap(colors_separate)


# functions for plotting masks, bboxes, images and alike
def show_mask(mask, ax, random_color=False):
    """
    Interpreted from https://github.com/facebookresearch/segment-anything
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=300):
    """
    Interpreted from https://github.com/facebookresearch/segment-anything
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
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


def crop_xyxy(img, mask, box, crop_box):
    """
    Input:
        PIL image, OR ndarray image,
        np.array mask,
        box in format x0,y0,x1,y1,
        x0,y0 is the left upper corner of the crop
        x1,y1 is the left upper corner of the crop

    Description:
    Crop the image mask and bounding box, starting at coords x0,y0 at the left upper corner
    w,h sets the size of the resulting window
    """
    img = utils.to_plt(img)
    crop_box = np.int32(crop_box)  # to int
    x0, y0, x1, y1 = crop_box[0], crop_box[1], crop_box[2], crop_box[3]

    if img.__class__.__name__ == "Image":  # PIL image
        cropped_img = img.crop((x0, y0, x1, y1))
    elif img.__class__.__name__ == "ndarray":  # numpy array
        cropped_img = img[y0:y1, x0:x1]
    elif img.__class__.__name__ == "Tensor":  # torch tensor??
        cropped_img = img[y0:y1, x0:x1]
    else:
        raise ValueError("Unknown image type")

    box_coords = [
        box[0] - x0,
        box[1] - y0,
        box[2] - x0,
        box[3] - y0,
    ]  # subtract corner from the box
    cropped_mask = mask[y0:y1, x0:x1]  # need to also crop the w,h

    return utils.to_torch_order(cropped_img), cropped_mask, box_coords
