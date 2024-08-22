# Detection and segmentation for FG-BG image decomposition

This repository is for exploration of different object detection and segmentation approaches.


### Main pipeline:
    Datasets
        COCO - Put in Datasets/COCO
    Detection 
        MViT https://github.com/mmaaz60/mvits_for_class_agnostic_od.git 
        PROB https://github.com/orrzohar/PROB.git
        Uni-Detector (?)
    Segmentation
        SAM-1 (downloaded by pip)

### Code structure, main modules:
    evaluator.py - main pipeline class. when run directly, evaluates the pipeline
    sam.ipynb - sam sequential inference on images, visuals of masks etc.
    coco_visuals.ipynb - notebook working with coco dataset
    IoU_recall_visuals - loads saved IoU values and produces histograms etc.
    test_all.py - runs all available tests for the code
    
    /detection_models - detection model wrappers, dummy classes
        dummy_detectors.py - basic detector wrapper structure 
            + 2 *dummy* classes, which produce GT boxes and GT+ middle point
        mvit_wrapper.py - wrapper for MViT model - *Not finished*, might not be used at all
        prob_wrapper.py - wrapper for PROB model - *Not finished*, might not be used at all
    
    /segmentation_models - segmentation model wrappers 
        base_seg_wrapper - basic segmentation wrapper structure 
        sam1_wrapper.py - 2 classes: SamWrapper and AutomaticSam, sam loading function
        sam2_wrapper.py - Not implemented yet, will ad SAM-2

    /utils
        utils.py - general utility functions. work with bboxes and masks
        jupyter_utils.py - code snippets used for data visualisation in plt. 

    /testing
        cuda_test.py - testing cuda availability and where CUDA_HOME is set to

    /datasets - where to put datasets files and files working with the datasets
        dataset_loading.py - coco wrapper, filepath parsing, test for coco_loading

    /config - (not used right now, will use config files in the future)


### Needed coco structure (for year=2017, analogous for 2014 in the same place)
        COCO/
            train2017
            val2017
            test2017
            annotations/
                instances_train2017.json
                instances_val2017.json
                image_info_test2017.json


### To run:
    Refer to install.md for installation recommendations and useful commands
    After installing:
        Test everything loading by running:
            python test_all.py
        
        run python evaluator.py, change pipeline parameters in main
        coco_visuals.ipynb, sam.ipynb contain visualisation of respective functionality
    

### Important links:
DETR - how to get the attention points?
    https://github.com/facebookresearch/detr/issues/593



