import utils
import numpy as np


def iou_against_one_box(one, multiple_boxes):
    # one box and a list against it
    return [utils.get_IoU_boxes(one, box) for box in multiple_boxes]


def iou_against_one_mask(one, multiple_masks):
    return [utils.get_IoU_masks(masks, mask) for mask in gt_masks]


def matching_fn(gt, det, det_scores, threshold=0.5, iou_type="boxes"):
    """
    TODO: docstring, input types, explain better
    """
    # go through DT (in order of confidence)
    # IoU vs all GT
    # match to highest IoU GT
    # remove this GT from list
    # if below threshold (?) ignore detections
    # if no GT, ignore detections
    # return DT for every GT in the list
    # return IoUs with the matches :))

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    assert len(det) == len(
        det_scores
    ), "Detection boxes and scores must have the same length"
    assert threshold >= 0, "Threshold must be greater or equal to 0"
    assert threshold <= 1, "Threshold must be smaller or equal to 1"

    match_indices = -np.ones(len(gt), dtype=np.int32)  # -1 if no match
    match_ious = np.zeros(len(gt), dtype=np.float32)  # 0 if no match
    gt_list = gt.copy()
    det_list = np.array(det.copy())

    # sort the detections according to their scores TODO: check order? ascending?
    indices_detections = np.argsort(det_scores)[::-1]  # from highest
    det_list = det_list[indices_detections]

    for det_index, dt_obj in enumerate(det_list):
        # calculate IoU against all GT boxes in the list
        if iou_type == "boxes":
            ious = iou_against_one_box(one=dt_obj, multiple_boxes=gt_list)
        elif iou_type == "masks":
            ious = iou_against_one_mask(one=dt_obj, multiple_masks=gt_list)
        # each time the list is smaller, so we dont match twice
        print(ious)
        # find highest IoU index (which GT is the best?)
        index_highest = np.argmax(ious)
        iou_best = ious[index_highest]
        gt_best = gt_list[index_highest]

        if iou_best <= threshold:
            continue  # remaining IoUs are too low, no match here

        # save which det box is matched with this GT, save IoU
        index_of_gt = gt.index(gt_best)
        match_indices[index_of_gt] = det_index
        match_ious[index_of_gt] = iou_best

        gt_list.pop(index_of_gt)  # GT is matched, remove from list

    # translate indices in detections for each gt into boxes
    matched_objects = [det_list[i] if i != -1 else None for i in match_indices]
    return matched_objects, match_ious


def test_matching_fn():
    # define dummy data
    gt_boxes = [[10, 10, 50, 50], [30, 30, 70, 70], [60, 60, 100, 100]]
    dt_boxes = [
        [0, 0, 50, 50],
        [0, 0, 80, 80],
        [30, 70, 70, 80],
        [90, 90, 130, 130],
        [10, 10, 50, 50],
        [100, 100, 140, 140],
        [0, 20, 30, 40],
        [40, 40, 80, 1000],
        [70, 70, 50, 110],
    ]

    dt_scores = [0.8, 0.6, 0.9, 0.7, 0.1, 0.2, 0.3, 0.4, 0.5]

    # Call the matching function
    matched_objects, match_ious = matching_fn(
        gt_boxes, dt_boxes, dt_scores, threshold=0.5
    )

    # Print the results
    print("Matched Objects:")
    for i, obj in enumerate(matched_objects):
        print(f"GT Box {i+1}: {obj}")

    print("Match IoUs:")
    for i, iou in enumerate(match_ious):
        print(f"GT Box {i+1}: {iou}")


# Run the test function
if __name__ == "__main__":
    test_matching_fn()
