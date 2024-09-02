import utils
import numpy as np
import torch


def iou_against_one_box(one, multiple_boxes):
    # one box and a list against it
    return [utils.get_IoU_boxes(one, box) for box in multiple_boxes]


def iou_against_one_mask(one, multiple_masks):
    return [utils.get_IoU_masks(one, mask) for mask in multiple_masks]


def matching_fn(gt_full_array, det, det_scores, threshold=0.5, iou_type="boxes"):
    """
    1 to 1 matching of boxes or masks to gt list
    orders objects to be matched in order based on confidence score
    for one of these objects, IoU against gt (still not matched in a list) is computed
    we take the match with highest IoU, if it is above threshold
    we pop this gt out of the list and continue. if no gt is left, we exit

    returns:
        - matched_objects: list of matched objects, None if no match
        - match_ious: list of IoUs for each matched object, 0 if no match

    this function based on https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    """

    # input check , some type conversions
    det_scores = np.array(det_scores)
    assert len(det) == len(
        det_scores
    ), "Detection boxes and scores must have the same length"
    assert threshold >= 0, "Threshold must be greater or equal to 0"
    assert threshold <= 1, "Threshold must be smaller or equal to 1"
    assert (det_scores >= 0).all, "Detection scores must be greater or equal to 0"
    assert (det_scores <= 1).all, "Detection scores must be smaller or equal to 1"

    # arrays holding results, init as default
    match_indices = -np.ones(len(gt_full_array), dtype=np.int32)  # -1 if no match
    match_ious = np.zeros(len(gt_full_array), dtype=np.float32)  # 0 if no match

    # convert GT to list for pop-ing, detection & score array for sorting
    gt_list = list(gt_full_array)  # list for pop-ing! , need to remove matched GTs
    gt_full_array = np.array(gt_full_array)  # need index of something from gt
    det_full_array = np.array(det)  # array for sorting

    # sort the detections according to their scores
    indices_detections = np.argsort(det_scores)[::-1]  # from highest
    det_full_array = det_full_array[indices_detections]

    for det_index, detected_object in enumerate(det_full_array):

        if len(gt_list) <= 0:  # all GT matched :))
            break
        # calculate IoU against all GT boxes in the list
        if iou_type == "boxes":
            ious = iou_against_one_box(one=detected_object, multiple_boxes=gt_list)
        elif iou_type == "masks":
            detected_object = torch.Tensor(detected_object)
            ious = iou_against_one_mask(one=detected_object, multiple_masks=gt_list)
        # each time the list is smaller, so we can't match twice

        # find highest IoU index (which GT is the best?)
        index_highest = np.argmax(ious)
        iou_best = ious[index_highest]
        gt_best = gt_list[index_highest]

        if iou_best <= threshold:
            continue  # remaining IoUs are too low, no match for this detected box

        # save which det box is matched with this GT, save IoU
        if iou_type == "boxes":
            index_of_gt = np.where(gt_full_array == gt_best)[0][0]
        elif iou_type == "masks":
            matches = np.all(gt_full_array == np.array(gt_best), axis=(1, 2))
            index_of_gt = np.where(matches)[0][0]

        match_indices[index_of_gt] = det_index
        match_ious[index_of_gt] = iou_best

        gt_list.pop(index_highest)  # this GT is matched, remove from list

    # translate indices in det for each gt into boxes
    matched_objects = [det_full_array[i] if i != -1 else None for i in match_indices]
    return matched_objects, match_ious


def test_matching_fn():
    # define dummy data
    gt_boxes = [[10, 10, 50, 50], [60, 65, 100, 100], [30, 30, 70, 70]]
    dt_boxes = [
        [100, 100, 200, 300],
        [0, 0, 50, 50],
        [0, 0, 80, 80],
    ]
    dt_scores = [0.9, 0.8, 0.6]

    # Call the matching function
    matched_objects, match_ious = matching_fn(
        gt_boxes, dt_boxes, dt_scores, threshold=0.2
    )

    # Print the results
    print("Test 1: Threshold 0.2")
    print("Matched Objects:")
    for i, obj in enumerate(matched_objects):
        print(f"GT Box {i+1}: {obj}")

    print("Match IoUs:")
    for i, iou in enumerate(match_ious):
        print(f"GT Box {i+1}: {iou}")

    # second test, same as GT but permuted
    dt_boxes = [
        [30, 30, 70, 70],  # the same as gt, but permuted
        [10, 10, 50, 50],
        [60, 65, 100, 100],
        [0, 0, 1000, 1000],  # 2 really bad ones, one with high score
        [0, 0, 500, 500],
    ]
    dt_scores = [0.2, 0.8, 0.1, 0.9, 0.0]

    # Call the matching function
    matched_objects, match_ious = matching_fn(
        gt_boxes, dt_boxes, dt_scores, threshold=0.5
    )

    # Print the results
    print("\nTest 2: Threshold 0.5, same as GT and 2 dummy boxes")
    print("Matched Objects:")
    for i, obj in enumerate(matched_objects):
        print(f"GT Box {i+1}: {obj}")

    print("Match IoUs:")
    for i, iou in enumerate(match_ious):
        print(f"GT Box {i+1}: {iou}")
    # almost no boxes do have match bcause of the threshold


# Run the test function
if __name__ == "__main__":
    test_matching_fn()
