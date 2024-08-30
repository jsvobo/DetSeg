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

    matched_indices = -np.ones(len(gt), dtype=int)
    gt_list = gt.copy()
    dt_list = dt.copy()

    # sort the detections according to their scores TODO: check order
    indices_detections = np.argsort(dt_scores)
    dt_list = dt_list[indices_detections]

    for i, dt_box in enumerate(dt_list):
        # calculate IoU against all GT boxes in the list
        ious = iou_against_one_box(one=dt_box, multiple_boxes=gt_list)

        # sort IoUs and store GT order
        indices_IoU = np.argsort(ious)
        for iou in ious:
            if iou <= threshold:
                continue  # remaining IoUs are low

        max_iou_idx = np.argmax(ious)
        matched_indices[max_iou_idx] = i
        gt_list.pop(max_iou_idx)
