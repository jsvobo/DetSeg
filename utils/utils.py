import numpy as np

def get_coco_boxes(metadata):
    '''
    Takes the COCO metadata and returns the bounding boxes. 
    for one image
    '''
    boxes=[]
    for ann in metadata:
        box=box_coco_to_sam(ann['bbox'])
        boxes.append(box)
    return boxes

def get_coco_masks(metadata:dict,api_class):
    '''
    Takes the COCO metadata and returns the segmentation masks. 
    for one images
    '''
    masks=[]
    for ann in metadata:
        mask = api_class.annToMask(ann)
        masks.append(mask)

    return masks

def get_IoU_masks(gt_mask,mask):
    ''' 
    Compute IoU between 2 masks
    '''
    
    intersection = np.sum((gt_mask + mask)==2)
    union =np.sum((gt_mask + mask)!=0)  
    IoU = intersection/union

    return IoU

def box_coco_to_sam(coco_box):
    '''
    Convert coco box to sam box
    from x0,y0,w,h to x0,y0,x1,y1
    '''
    return coco_box[0],coco_box[1],coco_box[0]+coco_box[2],coco_box[1]+coco_box[3]

def get_middle_point(box):
    '''
    Get the middle point of a bounding box in format x0,y0,x1,y1
    '''
    x0, y0 = box[0], box[1]
    x1, y1 = box[2], box[3]
    return [(x0+x1)/2,(y0+y1)/2]







