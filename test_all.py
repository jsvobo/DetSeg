from testing.cuda_test import test_cuda
import datasets
from segmentation_models.sam1_wrapper import test_sam1_wrappers
import detection_models
from evaluator import test_evaluator
from matching import test_matching_fn

if __name__ == "__main__":
    test_cuda()
    datasets.test_coco_loading()
    datasets.test_imagenet_loading()
    test_sam1_wrappers()
    test_evaluator()
    test_matching_fn()
    test_grounding_dino_tiny()
