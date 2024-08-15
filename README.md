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
    sam.ipynb - testing coco loading
    test_all.py - runs all available tests for the code
    detection_pipeline.py - main pipeline logic
    coco_visuals.ipynb - notebook working with coco dataset
    
    /run_detection - files and classes working with external models
        run_mvit.py - wrapper for MViT model
        run_prob.py - wrapper for PROB model
    
    /utils
        utils.py - general utility functions. work with bboxes and masks
        sam_utils.py - code for wiking with sam model(s)
        visual_utils.py - code snippets used for data visualisation in plt

    /testing
        cuda_test.py - testing cuda availability and where CUDA_HOME is set to

    /datasets - where to put datasets files and files working with the datasets
        saver_loader.py - TODO: daving and loading partial results
        dataset_loading.py - coco wrapper, filepath parsing

    /config - (not used right now, will use config files in the future)

### Working with our packages:
    to import everything from utils:
        import utils
    then use utils.fn() or just fn() (since they are unpacked by *) to call any function from any utils package
    similarly with datasets 

    from run detection, you can import 2 folders at once: mvit and prob, which contain definitions for wrappers to the respective models
        import run_detection 
        then use mvit.detect_objects(...) and so on (cant use the function dorectly, since they are naming conflicts between the inferrence functions)

### Recommended coco structure
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
    
    Test coco loading by running:
        python datasets/dataset_loading.py 

    You can see coco images in coco_visuals.ipynb
    Eventually download PROB and MViT detection models  (not functional rn)
        git clone https://github.com/mmaaz60/mvits_for_class_agnostic_od.git 
        git clone https://github.com/orrzohar/PROB.git
    
### TODO:
    Sgementation and detection metrics
    Batching SAM on one image (all boxes)
    Detectors - Wrappers & get working
    Other datasets ?
    SAM-2 ?

### Important links:
DETR - how to get the attention points?
    https://github.com/facebookresearch/detr/issues/593




## Commands
        git clone https://github.com/mmaaz60/mvits_for_class_agnostic_od.git 
        git clone https://github.com/orrzohar/PROB.git

    make sure python interpreter sees the script location inside the projects (better use __init__.py)
        export PYTHONPATH=$(pwd)/mvits_for_class_agnostic_od:$PYTHONPATH

    CUDA_HOME should look like '/home.stud/svobo114/.conda/envs/detect_env'
    inbetween steps, feel free to check if CUDA is available in python and jupyter, or if CUDA_HOME is set to something sensible (need full toolkit)

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


