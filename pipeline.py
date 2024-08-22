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


segmentation_dict = {"sam1": segmentation_models.SamWrapper, "sam2": None}
detection_dict = {
    "gt boxes": detection_models.GTboxDetector,
    "gt boxes middle": detection_models.GTboxDetectorMiddle,
}
dataset_dict = {"coco": {"loader": CocoLoader, "split_fn": get_coco_split}}


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
        # later we will load from a file
        config_path = "./config/pipeline_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        mb = config["max_batch"]
        config["max_batch"] = None if mb == "None" else int(mb)
        config["save_results"] = bool(config["save_results"])
        config["print_results"] = bool(config["print_results"])
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
        sam_size = conf["segmentation"]["model"]
        segmentation_model = seg_class(device=device, model=sam_size)

        # additional functions in the pipeline. None for now
        boxes_transform = None  # no transformation for now
        box_matching = None  # selects boxes to use for seg.

        # metrics for det, seg batch and seg pairwise
        det_batch_metrics = MetricCollection(
            [
                # IntersectionOverUnion(),
                MeanAveragePrecision(
                    iou_type="bbox",
                    average="macro",
                    class_metrics=False,
                ),
            ]
        )
        seg_pairwise_metrics = MetricCollection(JaccardIndex("binary").to(device))
        seg_batch_metrics = MetricCollection(
            [
                # JaccardIndex("binary"),
                MeanAveragePrecision(
                    iou_type="segm",
                    average="macro",
                    class_metrics=False,
                ),
            ]
        )

        # Instantiate the Evaluator with the requested parameters
        evaluator = Evaluator(
            device=device,
            model_det=detector,
            model_seg=segmentation_model,
            det_batch_metrics=det_batch_metrics,
            seg_batch_metrics=seg_batch_metrics,
            seg_pairwise_metrics=seg_pairwise_metrics,
            boxes_transform=boxes_transform,
            box_matching=box_matching,
        )

        # loader from dataset
        data_loader = dataset.instantiate_loader(batch_size=batch_size, num_workers=4)

        # run the evaluation
        box_IoUs, mask_IoUs = evaluator.evaluate(data_loader, max_batch=max_batch)

        # print results
        if print_results:
            print("\n\n")
            print(det_batch_metrics.compute())
            print("\n")
            if segmentation_model is not None:
                print(seg_batch_metrics.compute())
                print("\n")
                print(seg_pairwise_metrics.compute())

        # process results further (mAP, Recall, IoU(s) etc.)
        # save results to file?
        if save_results:
            pass
            """ 
            print("Mean IoU: " + str(dataset_IoU.compute()))
            filename = f"./out/coco_base_thresholds_{len(thresholds)}.npy"
            # save to a file
            with open(filename, "wb") as f:
                np.save(f, np.array(thresholds))
            """


if __name__ == "__main__":
    pipeline = Pipeline()
    # runs the whole thing after loading it :))
