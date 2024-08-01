import torch
import os
import torchvision.datasets
from torchvision.datasets import CocoDetection


class DatasetCOCO():
    def __init__(self, datasets_dir="Datasets", dir="COCO", year="2017",type_ann="instances"):
        self.datasets_dir = datasets_dir
        self.dir = dir
        self.year = year
        self.type_ann= type_ann

        path_images = os.path.join(self.datasets_dir, self.dir)
        path_annotations = os.path.join(self.datasets_dir, self.dir, "annotations")

        self.train = os.path.join(path_images, "train" + self.year)
        self.test = os.path.join(path_images, "test" + self.year)
        self.val = os.path.join(path_images, "val" + self.year)
        self.ann_train = os.path.join(path_annotations, self.type_ann + "_train" + self.year + ".json")
        self.ann_val = os.path.join(path_annotations, self.type_ann + "_val" + self.year + ".json")
        self.ann_test = os.path.join(path_annotations, self.type_ann +"_test" + self.year + ".json")
    
    def load_train(self,transformations=None):
        dataset_train = torchvision.datasets.CocoDetection(
            root=self.train,
            annFile=self.ann_train,
            transform=transformations
        )

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda x: tuple(zip(*x))
        )

        return data_loader_train,dataset_train

if __name__ == "__main__": 
    torch.manual_seed(0)
    coco = DatasetCOCO()
    coco.load_train()

    print("Loaded")
    print('Number of samples: ', len(coco_train))