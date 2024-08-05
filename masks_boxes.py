
def calculate_iou(boxA, boxB):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    :param boxA: First bounding box [x1, y1, x2, y2].
    :param boxB: Second bounding box [x1, y1, x2, y2].
    :return: IoU value.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def compare_bounding_boxes(bboxes1, bboxes2):
    """
    Compare two sets of bounding boxes.
    
    :param bboxes1: First set of bounding boxes.
    :param bboxes2: Second set of bounding boxes.
    :return: List of IoU values.
    """
    iou_values = []
    for boxA in bboxes1:
        for boxB in bboxes2:
            iou = calculate_iou(boxA, boxB)
            iou_values.append(iou)
    return iou_values
