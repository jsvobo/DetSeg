from testing.cuda_test import test_cuda
from datasets.coco_wrapper import test_coco_loading
from datasets.imagenet_wrapper import test_imagenet_loading
from segmentation_models.sam1_wrapper import test_sam1_wrappers
from evaluator import test_evaluator
from matching import test_matching_fn
from detection_models.gdino_wrapper import test_grounding_dino_tiny
from detection_models.gdino_full import test_grounding_dino_full

if __name__ == "__main__":
    test_cuda()

    # datasets
    test_coco_loading()
    test_imagenet_loading()

    test_sam1_wrappers()
    test_evaluator()
    test_matching_fn()

    # detection models
    test_grounding_dino_tiny()  # will output a lot of 0s, the model cannot recognize classes easily.
    test_grounding_dino_full()  # this is repaired in gdino full
