import torch
import os
import torchvision.datasets
from torchvision.datasets import CocoDetection
import pycocotools


class CocoLoader():
    ''' 
    Class that handles coco dataset loading, works with the original coco API (pycocotools) and torch interface

    based on dataset directory, directory of the dataset itself, year (2014 or 2017) and type of annotation (instances, captions, person_keypoints),
    the DatasetCOCO class will extract the data and masks through the API class and torch interface

    returned instances: torch dataset, torch data loader, pycocotools API class (which is used for parsing masks and annotations)
    '''
    def __init__(self, config=None):
        if config == None:
            self.datasets_dir = "Datasets"
            self.subdir = "COCO"
            self.year = "2017"
            self.type_ann= "instances"
        else: 
            # TODO: parse config? maybe from a file? hydra?
            pass
        
        self._parse_paths()
        
    def _parse_paths(self):
        path_images = os.path.join(self.datasets_dir, self.subdir)
        path_annotations = os.path.join(self.datasets_dir, self.subdir, "annotations")

        self.train = os.path.join(path_images, "train" + self.year)
        self.test = os.path.join(path_images, "test" + self.year)
        self.val = os.path.join(path_images, "val" + self.year)
        
        self.ann_train = os.path.join(path_annotations, self.type_ann + "_train" + self.year + ".json")
        self.ann_val = os.path.join(path_annotations, self.type_ann + "_val" + self.year + ".json")
        self.ann_test = os.path.join(path_annotations, "image_info" +"_test" + self.year + ".json") 

        self.train_info = {"root": self.train, "annFile": self.ann_train}
        self.val_info = {"root": self.val, "annFile": self.ann_val}
        self.test_info = {"root": self.test, "annFile": self.ann_test}
    
    def _load(self,info:dict,transformations:list=None):
        '''
        info = {"root": path to the dataset, "annFile": path to the annotation file}  -> specifies the dataset and annotation file to load
        Loading the annotation two times. one time for torch interface and one time for pycocotools API class
        TODO: check if this is the best way to do it
        '''

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

    def load_train(self,transformations: list = None):
        """
        Loads the COCO train dataset.
        Args:
            transformations (list, optional): A list of transformations to apply to the dataset. Defaults to None.
        Returns:
            train data loader, dataset and the COCO API class.
        """
        return self._load(self.train_info, transformations)
        
    def load_val(self,transformations:list=None):
        """
        Loads the COCO validation dataset.
        Args:
            transformations (list, optional): A list of transformations to apply to the dataset. Defaults to None.
        Returns:
            val data loader, dataset and the COCO API class.
        """
        return self._load(self.val_info, transformations)

    def load_test(self,transformations:list=None):
        """
        Loads the COCO test dataset.
        Args:
            transformations (list, optional): A list of transformations to apply to the dataset. Defaults to None.
        Returns:
            test data loader, dataset and the COCO API class.
        """
        return self._load(self.test_info, transformations)


    def load_all(self,transformations: list = None):
        """
        Load the COCO dataset.
        Returns training, testing, and validation datasets, 
        as well as API classes for each of them in a dictionary.
        Args:
            transformations (list, optional): A list of transformations to apply to the dataset. Defaults to None.
        Returns:
            A dictionary containing the loaders, datasets, and API classes for each split.
        """

        train_loader, train_dataset, train_annotations = self.load_train(transformations)
        val_loader, val_dataset, val_annotations = self.load_val(transformations)
        test_loader, test_dataset, test_annotations = self.load_test(transformations)
        
        return {    
            'test': {'loader': test_loader, 'dataset': test_dataset, 'annotations': test_annotations},
            'train': {'loader': train_loader, 'dataset': train_dataset, 'annotations': train_annotations},
            'val': {'loader': val_loader, 'dataset': val_dataset, 'annotations': val_annotations}
        }
        


if __name__ == "__main__": 
    print("\nTesting coco loading")
    torch.manual_seed(0)
    coco = CocoLoader()

    loader, dataset, annotations = coco.load_train()
    print("\nLoaded train dataset")
    print('Number of samples: ', len(dataset))
    cats = annotations.loadCats(annotations.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('train categories: \n{}\n'.format(' '.join(nms)))

    loader, dataset, annotations = coco.load_val()
    print("\nLoaded val dataset")
    print('Number of samples: ', len(dataset))
    cats = annotations.loadCats(annotations.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('val categories: \n{}\n'.format(' '.join(nms)))

    loader, dataset, annotations = coco.load_test()
    print("\nLoaded test dataset")
    print('Number of samples: ', len(dataset))
    cats = annotations.loadCats(annotations.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('test categories: \n{}\n'.format(' '.join(nms)))

    coco.load_all()
    print("Tests passed\n")
