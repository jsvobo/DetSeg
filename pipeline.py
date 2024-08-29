import torch
import torchvision
import numpy as np
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pprint

import utils
import datasets.dataset_loading as datasets
import detection_models
import segmentation_models
from evaluator import Evaluator

from torchmetrics import MetricCollection
from torchmetrics.classification import JaccardIndex
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# dicts with implemented models, to specify class from reading confif as string


def hardcoded_config():
    # for testing purposes, dummy config.
    # change if the config structure changes, otherwise tests wont work
    return DictConfig(  # rewrite to load one special config for testing
        {
            "dataset": {"name": "coco", "split": "val", "year": 2017},
            "detector": {"name": "gt_boxes_middle"},
            "segmentation": {"name": "sam1", "model": "b"},
            "save_results": False,
            "print_results": True,
            "batchsize": 6,
            "max_batch": 2,  # early cutoff for testing
            "save_path": "./out/pipeline_results/",
        }
    )


class Pipeline:
    def __init__(self, cfg: DictConfig):
        cfg_parsed = self.prepar_cfg(cfg)
        self.run(cfg_parsed)

    def prepare_dataset(self, cfg: DictConfig):
        # prepare dataset from config
        split = cfg.split
        year = cfg.year
        root = cfg.root

        dataset_class = cfg.class_name
        get_path = cfg.split_fn

        path = getattr(datasets, get_path)(split=split, year=year, root=root)
        dataset = getattr(datasets, dataset_class)(path, transform=None)
        return dataset

    def prepare_det(self, cfg: DictConfig):
        # prepare detection model based on config
        model_name = cfg.class_name
        if model_name == "None":
            model_det = detection_models.GTboxDetector()
        else:
            model_det = getattr(detection_models, model_name)()
        return model_det

    def prepare_seg(self, cfg: DictConfig, device: str):
        # prepare segmentation model based on config
        print(cfg)
        model_name = cfg.class_name
        if model_name == "None":
            segmentation_model = None
        else:
            segmentation_model = getattr(segmentation_models, model_name)(
                device=device, cfg=cfg.sam_model
            )
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

        print("\nResults:")
        print(f"Mean segmentation IoU: {round(float(mean_seg_IoU),3)}")
        print(f"Weighted segmentation IoU: {round(float(weighed_seg_IoU),3)}")

        pprint.pprint(
            result_dict["classless mAP - segmentation"]
        )  # add more dicts here?

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
        boxes_transform = None  # no transformation for now.

        # Instantiate the Evaluator with the requested parameters
        evaluator = Evaluator(
            device=device,
            model_det=detector,
            model_seg=segmentation_model,
            boxes_transform=boxes_transform,
        )

        # loader from dataset
        data_loader = dataset.instantiate_loader(
            batch_size=cfg.batch_size, num_workers=4
        )

        # run the evaluation
        evaluator.evaluate(data_loader, max_batch=cfg.max_batch)
        result_dict, array_masks, array_boxes = (
            evaluator.get_metrics()
        )  # collect results

        # print results
        if print_results:
            self.print_results(result_dict)

        # save result dictionary and metadata to a file.
        if save_results:
            utils.save_results(
                result_dict=result_dict,
                array_masks=array_masks,
                array_boxes=array_boxes,
                cfg=cfg,
            )


@hydra.main(config_path="config", config_name="pipeline_config.yaml", version_base=None)
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    pipeline = Pipeline(cfg)


def test_pipeline():
    print("\nTesting pipeline")
    cfg = hardcoded_config()
    pipeline = Pipeline(cfg)  # run on coco
    print("Pipeline test passed!")


if __name__ == "__main__":
    main()
