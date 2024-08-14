from datasets.dataset_loading import CocoLoader
import utils

import torch
import numpy as np
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader

# sam
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide  # for batching


batch_size = 5  # 5-10_500
num_workers = 4
shuffle = True
batch_max = 8_500


# load coco
coco = CocoLoader()
transforms = None  # No augmentations for now
data_train, api = coco.load_train(transformations=transforms)

# load segmentation model(b)
assert torch.cuda.is_available()
predictor, sam = utils.prepare_sam("cuda", model="b")
resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

data_loader = DataLoader(
    data_train,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    collate_fn=lambda x: tuple(zip(*x)),
)

# metrics:
dataset_IoU = JaccardIndex(task="binary")
thresholds = []

for i, batch in enumerate(data_loader):
    if i % 100 == 0:
        print("Batch: " + str(i))
    images_pil = list(batch[0])
    metadata = list(batch[1])

    # separate GT for metrics
    gt_boxes = []
    gt_masks = []
    images_to_process = []
    for j in range(len(images_pil)):
        masks_img, boxes_img = utils.coco_masks_boxes(
            metadata[j], api
        )  # load boxes x0,y0,x1,y1
        if len(boxes_img) == 0:  # no boxes in image
            continue

        boxes_img = torch.Tensor(boxes_img)  # change format and to tensor
        masks_img = torch.Tensor(masks_img)
        gt_boxes.append(boxes_img)
        gt_masks.append(masks_img)
        images_to_process.append(images_pil[j])

    if len(gt_masks) == 0:  # all imgs in a batch with no masks
        print("No masks in batch")
        continue

    inferrence_boxes = gt_boxes

    # prepare input for batch
    sam_batched_inputs = []
    for j in range(len(images_to_process)):
        img = np.array(images_to_process[j])
        dict_img = {  # written according to official sam notebook predictor.ipynb
            "image": utils.prepare_image_for_batch(img, resize_transform, sam.device),
            "boxes": resize_transform.apply_boxes_torch(
                inferrence_boxes[j].to(sam.device), img.shape[:2]
            ),
            "original_size": img.shape[:2],
        }
        sam_batched_inputs.append(dict_img)

    # run inference
    batched_output = sam(sam_batched_inputs, multimask_output=True)

    # Take best masks in each image
    for j, dict_output in enumerate(batched_output):
        pred_quality = dict_output["iou_predictions"]
        best = np.argmax(pred_quality.cpu(), axis=1)

        arange = torch.arange(best.shape[0])
        best_masks = dict_output["masks"][arange, best]  # take best mask for each box

        dataset_IoU.update(best_masks.cpu(), torch.Tensor(gt_masks[j]))  # both on cpu?
        # calculate overlap with GT mask
        # save to thresholds
        overlaps = utils.get_IoU_multiple(best_masks.cpu(), gt_masks[j])
        thresholds.extend(overlaps)

    if i >= batch_max:  # just few batches for now, print final IoU
        print("Mean IoU: " + str(dataset_IoU.compute()))

        # flatten and to .npy, save to file

        filename = f"./out/coco_thresholds_{len(thresholds)}.npy"
        # save to a file
        with open(filename, "wb") as f:
            np.save(f, np.array(thresholds))
        break
