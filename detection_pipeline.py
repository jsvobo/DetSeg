
from dataset_loading import CocoLoader
import torch
import numpy as np

#sam
import sam_utils #sam_utils in the same directory
from segment_anything import SamPredictor, sam_model_registry

#others
import utils


'''
Main pipeline logic. 
load datasets
load detection models
load segmentation models

test detection on dataset, comapre with GT bboxes
test segmentation on dataset, compare with GT masks

Goal: 
    - metrics for detection
    - metrics for segmentation
    - output tables, output visualizations, saving BBoxes, masks, etc.
    - eventually another script to load the results without rerunning
    - reproducible code here, examples of usage
    - validation and training sets (just inference here)
'''
if __name__ == "__main__": 
    assert(torch.cuda.is_available())

    threshold=0.8 #load? or like a argument to the script?

    #load COCO dataset
    coco = CocoLoader()
    transforms = None #No augmentations for now
    data_train, api = coco.load_train(transformations=transforms)

    #load detection model(s) 
        #TODO ...
    
    #load segmentation model(s)
    predictor,sam = sam_utils.prepare_sam("cuda")
    IoU_avg=0
    successful_boxes=0

    for number_img,item in enumerate(data_train):
        if number_img>10: break #just testing here

        img = item[0] 
        metadata = item[1]

        #get GT
        gt_bboxes= utils.get_coco_boxes(metadata)
        gt_masks = utils.get_coco_masks(metadata,api)

        #setup SAM-1
        img=np.asarray(img)
        predictor.set_image(img) #sam

        #for GT boxes RN, else detection HERE!! TODO: detection module(s)
        detected_boxes = gt_bboxes 
            #TODO: metrics for detection against GT

        #Segmentation
        detected_masks=[] #mask for each box
        success=np.zeros(len(detected_boxes)) #1 if the mask was "good"

        for i,box in enumerate(detected_boxes): #prompt by box only
            masks, scores, logits = predictor.predict( #TODO: batched prediction? for all boxes at once?
                point_coords=None,
                point_labels=None,
                box=np.array(box),
                multimask_output=True, 
                )
            mask,score=sam_utils.select_best_mask(masks,scores)

            if score>=threshold:
                success[i]=1
                detected_masks.append(mask) 

                IoU = utils.compute_iou_one_img_masks(gt_masks[i],mask) #segmentation metrics
                print("IoU: ",IoU)
                IoU_avg+=IoU

        detected_masks=np.array(detected_masks)
        detected_boxes=np.array(detected_boxes)[success==1]
        successful_boxes+=len(detected_boxes)

        #TODO: metrics for segmentation against GT
        
    #aggregate metrics over whole dataset
    print("IoU avg: ",IoU_avg/successful_boxes) #detected boxes after filtering bad masks
    print("DONE")
    
