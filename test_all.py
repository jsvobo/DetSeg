from testing.cuda_test import test_cuda
from datasets.dataset_loading import test_coco_loading

if __name__ == "__main__": 
    ''' 
    test pipeline
    test loading all datasets
    test cuda
    test loading/saving?
    test everything I can?
    '''

    test_cuda()
    test_coco_loading()
    
