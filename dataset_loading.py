import torch
import os
import torchvision.datasets
from torchvision.datasets import CocoDetection
import pycocotools


class DatasetCOCO():
    #you need to have pycocotools installed (coco API)

    def __init__(self, datasets_dir="Datasets", dir="COCO", year="2017",type_ann="instances"):
        self.datasets_dir = datasets_dir
        self.dir = dir
        self.year = year
        self.type_ann= type_ann # instances, captions, person_keypoints

        path_images = os.path.join(self.datasets_dir, self.dir)
        path_annotations = os.path.join(self.datasets_dir, self.dir, "annotations")

        self.train = os.path.join(path_images, "train" + self.year)
        self.test = os.path.join(path_images, "test" + self.year)
        self.val = os.path.join(path_images, "val" + self.year)
        
        self.ann_train = os.path.join(path_annotations, self.type_ann + "_train" + self.year + ".json")
        self.ann_val = os.path.join(path_annotations, self.type_ann + "_val" + self.year + ".json")
        self.ann_test = os.path.join(path_annotations, self.type_ann +"_test" + self.year + ".json") #TODO: Check if this is correct

        self.train_info = {"root": self.train, "annFile": self.ann_train}
        self.val_info = {"root": self.val, "annFile": self.ann_val}
        self.test_info = {"root": self.test, "annFile": self.ann_test}
    
    def load(self,info,transformations=None):
        dataset = torchvision.datasets.CocoDetection( #torch interface class
            root=info["root"],
            annFile=info["annFile"],
            transform=transformations
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4,
        )

        api_class = pycocotools.coco.COCO(info["annFile"])  
        categories = api_class.loadCats(api_class.getCatIds())
        category_names = [category["name"] for category in categories]

        return loader, dataset, api_class
    
    def load_train(self, transformations=None):
        '''
        loads coco training dataset.
        returns:
            torch data loader
            dataset
            api class from coco API 
        '''
        return self.load(self.train_info, transformations)

    def load_val(self, transformations=None):
        '''
        loads coco validation dataset.
        returns:
            torch data loader
            dataset
            api class from coco API 
        '''
        return self.load(self.val_info, transformations)

    def load_test(self, transformations=None):
        '''
        loads coco testing dataset.
        returns:
            torch data loader
            dataset
            api class from coco API 
        '''
    
        return self.load(self.test_info, transformations)
    



if __name__ == "__main__": 
    print("Testing coco loading")
    torch.manual_seed(0)
    coco = DatasetCOCO()


    loader, dataset, annotations = coco.load_train()
    print("Loaded train dataset")
    print('Number of samples: ', len(dataset))
    cats = annotations.loadCats(annotations.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('train categories: \n{}\n'.format(' '.join(nms)))


    loader, dataset, annotations = coco.load_val()
    print("Loaded val dataset")
    print('Number of samples: ', len(dataset))
    cats = annotations.loadCats(annotations.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('val categories: \n{}\n'.format(' '.join(nms)))


    loader, dataset, annotations = coco.load_test()
    print("Loaded test dataset")
    print('Number of samples: ', len(dataset))
    cats = annotations.loadCats(annotations.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('test categories: \n{}\n'.format(' '.join(nms)))