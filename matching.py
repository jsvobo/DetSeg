import utils


def iou_against_one_box(one, multiple_boxes):
    # one box and a list against it
    results = [utils.get_IoU_boxes(one, box) for box in multiple_boxes]


def matching_fn(gt, dt, dt_scores, threshold=0.5):
    # go through DT (in order of confidence)
    # IoU vs all GT
    # match to highest IoU GT
    # remove this GT from list
    # if below threshold (?) ignore detections
    # if no GT, ignore detections
    # return DT for every GT in the list
    # return IoUs with the matches :))

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

    match_indices = -np.ones(len(gt), dtype=np.int32)
    match_IoUs = np.zeros(len(gt), dtype=np.float32)
    gt_list = gt.copy()
    dt_list = dt.copy()

    # sort the detections according to their scores TODO: check order
    indices_detections = np.argsort(dt_scores)
    dt_list = dt_list[indices_detections]

    for dt_box in dt_list:
        # calculate IoU against all GT boxes in the list
        ious = iou_against_one_box(one=dt_box, multiple_boxes=gt_list)
        # each time the list is smaller, so we dont match twice

        # find highest IoU index (which GT is the best?)
        index_highest = np.argmax(ious)
        iou_best = ious[index_highest]
        gt_best = gt_list[index_highest]

        if iou_best <= threshold:
            continue  # remaining IoUs are too low, no match here

        index_of_gt = gt.index(gt_best)
        match_indices[index_of_gt] = dt_box  # TODO add score?
        match_IoUs[index_of_gt] = iou_best

        gt_list.pop(index_of_gt)  # GT is matched, remove from list

    return match_indices, match_IoUs
