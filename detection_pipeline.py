
from dataset_loading import CocoLoader

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
'''
if __name__ == "__main__": 
    #can currently be seen in seg.ipynb

    coco = CocoLoader()
    transforms = None
    data_train, api = coco.load_train(transformations=transforms)
    #data_val, api = coco.load_val(transformations=transforms) 

    #load detection model(s)

    # for a batch
        #get bboxes, produce other bboxes via models
    
        #compare bboxes

    #metrics
    
