# Detection and segmentation for FG-BG image decomposition

This repository is for exploration of different object detection and segmentation approaches.


### Main pipeline:
    Datasets
        COCO 
        Imagenet
    Detection 
        Grounding DINO (tiny, normal)
        not using detection (take GT boxes)
    Segmentation
        SAM-1 (small, huge)
        No segmentation, only detection

### Code structure, main modules:
    pipeline.py - main pipeline class. when run directly, evaluates the pipeline loaded from config
    evaluator.py - contains evaluator class used in the pipeline. can be also run directly
    loading_imagenet.py - special loader for results on imagenet + image with gt class. 

    sam.ipynb - sam sequential inference on images, visuals of masks etc.
    coco_visuals.ipynb - notebook working with coco dataset
    IoU_recall_visuals.ipynb - loads saved IoU values and other results and produces histograms etc.
    test_all.py - runs all available tests for the code
    
    /detection_models - detection model wrappers, dummy classes
        dummy_detectors.py - basic detector wrapper structure 
            + 2 *dummy* classes, which produce GT boxes and GT+ middle point
        gdino_full - wrapper for gdino full
        gdino_wrapper - base class for gdino + gdino tiny from huggingface wrapper. class indices search is not working

    /segmentation_models - segmentation model wrappers 
        base_seg_wrapper - basic segmentation wrapper structure 
        sam1_wrapper.py - 2 classes: SamWrapper and AutomaticSam, sam loading function


    /utils
        utils.py - general utility functions. work with bboxes and masks
        df_utils.py - utils used for analysis of dataframes. for example in IoU_recall_visuals.ipynb and imagenet_fun.ipynb
        visual_utils.py - code snippets used for data visualisation in plt. 
        io_utils.py -support code for saving/loading data

    /testing
        cuda_test.py - testing cuda availability and where CUDA_HOME is set to

    /datasets - where to put datasets files and files working with the datasets
        coco_wrapper.py - coco wrapper, filepath parsing, test for coco_loading
        imagenet_wrapper.py - similar wrapper, but for imagenet dataset
        saver_loader.py - classes for saving and then loading results. saver is passed into evaluator by the pipeline class

    /config 
        pipeline_config.yaml - main config for the pipeline. (the project uses hydra to manage configurations)
        other folders - config structure used by hydra. refer to the subfolders for details


### To run:
    Refer to install.md for installation recommendations and useful commands
    After installing:
        Test everything loading by running:
            python test_all.py
        
        run python pipeline.py, change parameters in the config/pipeline_config.yaml and other configs
        results of the pipeline are saved locally, can be loaded in IoU_recall_visuals.ipynb notebook
    
    To run from cmd: 
        python pipeline.py detector.class_name=None max_batch=10 batch_size=1 segmentation.class_name=None detector.class_name=None 

### Important links:
DETR - how to get the attention points?
    https://github.com/facebookresearch/detr/issues/593



