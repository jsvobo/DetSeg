
from dataset_loading import CocoLoader


if __name__ == "__main__": 
    #can currently be seen in seg.ipynb

    coco = CocoLoader()
    transforms = None
    loader_train, data_train, api = coco.load_train(transformations=transforms)
    loader_val, data_val, api = coco.load_val(transformations=transforms) 

    #load detection model(s)

    # for a batch
        #get bboxes, produce other bboxes via models
    
        #compare bboxes

    #metrics
    
