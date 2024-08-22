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


### Needed coco structure (for year=2017, analogus for 2014 in the same place)
        COCO/
            train2017
            val2017
            test2017
            annotations/
                instances_train2017.json
                instances_val2017.json
                image_info_test2017.json


### To run:
    Install requirements and setup conda env.:
        conda create -n detect_env 
        conda activate detect_env
        conda install python=3.11 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
        pip install -r ./requirements.txt 


    Copy/link coco to datasets/ folder ()
        cp location ./datasets/
        ln -s location ./datasets/
    
    Test everything loading by running:
        python test_all.py
    
    run python evaluator.py, change pipeline parameters in main
    coco_visuals.ipynb, sam.ipynb contain visualisation of respective functionality
    

### Important links:
DETR - how to get the attention points?
    https://github.com/facebookresearch/detr/issues/593



## Commands - Collection of useful commands not needed rn, keep here for future reference

    Eventually download PROB and MViT detection models  (not functional rn)
        git clone https://github.com/mmaaz60/mvits_for_class_agnostic_od.git 
        git clone https://github.com/orrzohar/PROB.git


    make sure python interpreter sees the script location inside the projects (better use __init__.py)
        export PYTHONPATH=$(pwd)/mvits_for_class_agnostic_od:$PYTHONPATH

    CUDA_HOME should look like '/home.stud/svobo114/.conda/envs/detect_env'
    in-between steps, feel free to check if CUDA is available in python and jupyter, or if CUDA_HOME is set to something sensible (need full toolkit) or set manually

        conda create -n detect_env 
        conda activate detect_env
        conda install python=3.11 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
        pip install -r ./requirements.txt 
        conda install nvidia/label/cuda-11.8.0::cuda-toolkit


    you should see True and a path (if installed cuda-toolkit)
        python ./testing/cuda_test.py 

    build rust compiler in your shell if needed (for some transformers versions)
    When prompted, proceed with default installation
        --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh 

    build cuda operators for PROB as per https://github.com/orrzohar/PROB (start from DetSeg main dir)
        cd ./PROB/models/ops/
        sh ./make.sh

    build deformable attention modules in mvits as per https://github.com/mmaaz60/mvits_for_class_agnostic_od.git (start from DetSeg main dir)
        cd ./mvits_for_class_agnostic_od/models/ops
        sh ./make.sh 
    you need the cuda toolkit for this! -__-
    We have gcc 12, need<=11


    supply a backbone tro PROB model (start from DetSeg main directory)
        ln -s /mnt/vrg2/imdec/models/detectors/dino_resnet50_pretrain.pth PROB/models/

    mvit model should theoretically be run like: (but saves the results next to the data)
        python inference/main.py -m "$MODEL_NAME" -i "$DATASET_BASE_DIR/coco/val2017" -c "$CHECKPOINTS_PATH"

    sam
        pip install git+https://github.com/facebookresearch/segment-anything.git


