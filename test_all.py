from testing.cuda_test import test_cuda
from datasets.dataset_loading import test_coco_loading
from segmentation_models.sam1_wrapper import test_sam_wrappers
from evaluator import test_evaluator
from pipeline import test_pipeline


if __name__ == "__main__":
    test_cuda()
    test_coco_loading()
    test_sam_wrappers()
    test_evaluator()
    test_pipeline()
