import torch
import torchvision
import numpy as np
from evaluator import Evaluator

import utils
from datasets.dataset_loading import CocoLoader, get_coco_split
import segmentation_models
import detection_models

from torchmetrics import MetricCollection
from torchmetrics.classification import JaccardIndex
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import yaml

# dicts with implemented models, to specify class from reading confif as string
segmentation_dict = {"sam1": segmentation_models.SamWrapper, "sam2": None, "None": None}
detection_dict = {
    "gt boxes": detection_models.GTboxDetector,
    "gt boxes middle": detection_models.GTboxDetectorMiddle,
    "None": None,
}
dataset_dict = {
    "coco": {"loader": CocoLoader, "split_fn": get_coco_split},
}


class Pipeline:
    def __init__(self):
        conf = self.load_config()
        self.run(conf)

    def hardcoded_config(self):
        # for testing purposes
        return {
            "max_batch": 10,  # early cutoff for testing
            "dataset": {"name": "coco", "split": "val", "year": 2017},
            "detector": "gt boxes middle",
            "segmentation": {"name": "sam1", "batchsize": 6, "model": "b"},
            "save_results": False,
            "print_results": True,
            "save_path": "./out/",
        }

    def load_config(self):
        # TODO: add hydra
        config_path = "./config/pipeline_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        mb = config["max_batch"]
        config["max_batch"] = None if mb == "None" else int(mb)
        config["save_results"] = True if config["save_results"] == "True" else False
        config["print_results"] = True if config["print_results"] == "True" else False
        print(config["print_results"])
        return config

    def run(self, conf: dict):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = conf["segmentation"]["batchsize"]

        max_batch = conf["max_batch"]
        save_results = conf["save_results"]
        print_results = conf["print_results"]

        # load detector
        detector = detection_dict[conf["detector"]]()

        # load dataset
        dataset_name = conf["dataset"]["name"]
        dataset_class = dataset_dict[dataset_name]["loader"]
        get_path = dataset_dict[dataset_name]["split_fn"]
        split = conf["dataset"]["split"]
        year = str(conf["dataset"]["year"])
        dataset = dataset_class(get_path(split=split, year=year), transform=None)

        # load segmentation model
        seg_class = segmentation_dict[conf["segmentation"]["name"]]
        if seg_class is None:
            segmentation_model = None
        else:
            sam_size = conf["segmentation"]["model"]
            segmentation_model = seg_class(device=device, model=sam_size)

        # additional functions in the pipeline. None for now
        boxes_transform = None  # no transformation for now

        # Instantiate the Evaluator with the requested parameters
        evaluator = Evaluator(
            device=device,
            model_det=detector,
            model_seg=segmentation_model,
            boxes_transform=boxes_transform,  # TODO: pass cfg parts here
        )

        # loader from dataset
        data_loader = dataset.instantiate_loader(batch_size=batch_size, num_workers=4)

        # run the evaluation
        evaluator.evaluate(data_loader, max_batch=max_batch)
        result_dict = evaluator.get_metrics()

        # print results
        if print_results:
            mean_seg_IoU = result_dict["mean seg IoU"]
            weighed_seg_IoU = result_dict["weighted seg IoU"]
            print(f"Mean segmentation IoU: {round(float(mean_seg_IoU),3)}")
            print(f"Weighted segmentation IoU: {round(float(weighed_seg_IoU),3)}")
            # TODO: restructure this to print results from the output dict, just some? prune results in evaluator?

        # save result dictionary
        if save_results:
            # load output dir from config
            # TODO: save to a file with everything, then parse it? class/helper functions?
            pass
            """ 
            print("Mean IoU: " + str(dataset_IoU.compute()))
            filename = f"./out/coco_base_thresholds_{len(thresholds)}.npy"
            # save to a file
            with open(filename, "wb") as f:
                np.save(f, np.array(thresholds))
            """


if __name__ == "__main__":
    # runs the whole thing after loading it :))
    pipeline = Pipeline()
