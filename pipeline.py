import torch
import torchvision
import numpy as np
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

import utils
from datasets.dataset_loading import CocoLoader, get_coco_split
import segmentation_models
import detection_models
from evaluator import Evaluator

from torchmetrics import MetricCollection
from torchmetrics.classification import JaccardIndex
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# dicts with implemented models, to specify class from reading confif as string
segmentation_dict = {"sam1": segmentation_models.SamWrapper, "sam2": None, "None": None}
detection_dict = {
    "gt boxes": detection_models.GTboxDetector,
    "gt boxes middle": detection_models.GTboxDetectorMiddle,
}
dataset_dict = {
    "coco": {"loader": CocoLoader, "split_fn": get_coco_split},
}


def hardcoded_config():
    # for testing purposes, dummy config
    return {
        "dataset": {"name": "coco", "split": "val", "year": 2017},
        "detector": {"name": "gt boxes middle"},
        "segmentation": {"name": "sam1", "model": "b"},
        "save_results": False,
        "print_results": True,
        "batchsize": 6,
        "max_batch": 10,  # early cutoff for testing
        "save_path": "./out/pipeline_results/",
    }


class Pipeline:

    def __init__(self, cfg: DictConfig):
        cfg_parsed = self.prepar_cfg(cfg)
        self.run(cfg_parsed)

    def prepare_dataset(self, cfg: DictConfig):
        dataset_name = cfg.name
        split = cfg.split
        year = cfg.year

        dataset_class = dataset_dict[dataset_name][
            "loader"
        ]  # from predefined dict above
        get_path = dataset_dict[dataset_name]["split_fn"]

        dataset = dataset_class(get_path(split=split, year=year), transform=None)
        return dataset

    def prepare_det(self, cfg: DictConfig):
        model_name = cfg.name
        if model_name == "None":
            model_det = None
        else:
            model_det = detection_dict[model_name]()
        return model_det

    def prepare_seg(self, cfg: DictConfig, device: str):
        # which class is for this model?
        seg_class = segmentation_dict[cfg.name]

        if seg_class is None:
            segmentation_model = None
        else:
            sam_size = cfg.model
            segmentation_model = seg_class(device=device, model=sam_size)
        return segmentation_model

    def prepar_cfg(self, cfg: DictConfig):
        # change some small thing, like types from the loaded config
        mb = cfg.max_batch
        cfg.max_batch = None if mb == "None" else int(mb)
        cfg.dataset.year = str(cfg.dataset.year)
        return cfg

    def print_results(self, result_dict):
        mean_seg_IoU = result_dict["mean seg IoU"]
        weighed_seg_IoU = result_dict["weighted seg IoU"]

        print(f"Mean segmentation IoU: {round(float(mean_seg_IoU),3)}")
        print(f"Weighted segmentation IoU: {round(float(weighed_seg_IoU),3)}")

        # TODO: restructure this to print results from the output dict, just some? prune results in evaluator?
        # print in a structured way? a table? save a table for tex?

    def run(self, cfg: dict):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        save_results = cfg.save_results
        print_results = cfg.print_results

        # load detector
        detector = self.prepare_det(cfg.detector)

        # load dataset
        dataset = self.prepare_dataset(cfg.dataset)

        # load segmentation model
        segmentation_model = self.prepare_seg(cfg.segmentation, device)

        # additional functions in the pipeline. None for now
        boxes_transform = None  # no transformation for now. will load through dict

        # Instantiate the Evaluator with the requested parameters
        evaluator = Evaluator(
            device=device,
            model_det=detector,
            model_seg=segmentation_model,
            boxes_transform=boxes_transform,
        )

        # loader from dataset
        data_loader = dataset.instantiate_loader(
            batch_size=cfg.batchsize, num_workers=4
        )

        # run the evaluation
        evaluator.evaluate(data_loader, max_batch=cfg.max_batch)
        result_dict = evaluator.get_metrics()  # collect results

        # print results
        if print_results:
            self.print_results(result_dict)

        # save result dictionary and metadata
        if save_results:
            # load output dir from config
            # TODO: save to a file with everything, then parse it? class/helper functions?
            pass
            # utils.save_results(result_dict, metadata_dict, cfg:DictConfig)
            """ 
            print("Mean IoU: " + str(dataset_IoU.compute()))
            filename = f"./out/coco_base_thresholds_{len(thresholds)}.npy"
            # save to a file
            with open(filename, "wb") as f:
                np.save(f, np.array(thresholds))
            """


@hydra.main(config_path="config", config_name="pipeline_config.yaml", version_base=None)
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    pipeline = Pipeline(cfg)


def test_pipeline():
    cfg = hardcoded_config()
    pipeline = Pipeline(cfg)


if __name__ == "__main__":
    main()
