import torch
import torchvision
import numpy as np
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import pprint

import utils
import datasets
import detection_models
import segmentation_models
from evaluator import Evaluator

from torchmetrics import MetricCollection
from torchmetrics.classification import JaccardIndex
from torchmetrics.detection.iou import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class Pipeline:
    def __init__(self, cfg: DictConfig):
        cfg_parsed = self.prepar_cfg(cfg)
        self.cfg = cfg_parsed
        self.run(cfg_parsed)

    def prepare_dataset(self, cfg: DictConfig):
        # prepare dataset from config
        split = cfg.split
        year = cfg.year
        root = cfg.root

        transforms = cfg.transforms
        if transforms != "None":  # get the specific transforms from datasets module
            transforms = getattr(datasets, transforms)()
        else:
            transforms = None

        dataset_class = cfg.class_name
        get_path = cfg.split_fn

        path = getattr(datasets, get_path)(split=split, year=year, root=root)
        dataset = getattr(datasets, dataset_class)(path, transform=transforms)
        return dataset

    def prepare_class_list(self, cfg: DictConfig, dataset):
        if cfg.name == "dataset_defaults":  # if not, alternative prompts are used
            all_classes = dataset.get_classes()
        else:
            all_classes = cfg.classes  # defined above, move to config?
        return all_classes

    def prepare_det(self, cfg: DictConfig, all_classes):
        # prepare detection model based on config
        model_name = cfg.class_name
        if model_name == "None":
            model_det = detection_models.GTboxDetector()
        else:
            model_det = getattr(detection_models, model_name)(
                cfg=cfg, all_classes=all_classes
            )
        return model_det

    def prepare_seg(self, cfg: DictConfig, device: str):
        # prepare segmentation model based on config
        model_name = cfg.class_name
        if model_name == "None":
            segmentation_model = None
        else:
            segmentation_model = getattr(segmentation_models, model_name)(
                device=device, model=cfg.sam_model, cfg=cfg
            )
        return segmentation_model

    def prepar_cfg(self, cfg: DictConfig):
        # change some small thing, like types from the loaded config
        mb = cfg.max_batch
        cfg.max_batch = None if mb == "None" else int(mb)
        if "year" in cfg.dataset.keys():
            cfg.dataset.year = str(cfg.dataset.year)
        else:
            cfg.dataset.year = "None"

        cfg.evaluator.save_results = (
            cfg.save_results
        )  # want to save detections per image
        return cfg

    def print_result_dict(self, result_dict):
        """
        Structured output of the results for det and seg metrics
        """
        det = result_dict["detection"]
        seg = result_dict["segmentation"]

        print("\nResults:")
        print("Detection metrics:")
        utils.print_for_task(det)

        if self.cfg.segmentation.class_name != "None":
            print("Segmentation metrics:")
            utils.print_for_task(seg)

    def run(self, cfg: dict):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        save_results = cfg.save_results
        print_results = cfg.print_results

        # load dataset
        dataset = self.prepare_dataset(cfg.dataset)
        # if classes are needed for detection (DINO etc.)

        all_classes = self.prepare_class_list(cfg.class_list, dataset=dataset)

        # load detector
        detector = self.prepare_det(cfg.detector, all_classes=all_classes)

        # load segmentation model
        segmentation_model = self.prepare_seg(cfg.segmentation, device)

        # additional functions in the pipeline. None for now
        boxes_transform = None
        # no transformation for now. load from config? prepare func for transform

        # saver class, works with the batches and overall results
        saver = datasets.ResultSaver(cfg)

        # Instantiate the Evaluator with the requested parameters
        evaluator = Evaluator(
            cfg=cfg.evaluator,
            saver=saver,
            device=device,
            model_det=detector,
            model_seg=segmentation_model,
            boxes_transform=boxes_transform,
        )

        # loader from dataset
        data_loader = dataset.create_dataloader(
            batch_size=cfg.batch_size, num_workers=4
        )

        # run the evaluation and collect results
        evaluator.evaluate(data_loader, max_batch=cfg.max_batch)
        results = evaluator.get_results()

        if print_results:  # print results if requested
            self.print_result_dict(results["metrics"])

        if (
            save_results
        ):  # save results using saver her. results per image are saved inside the evaluator
            saver.save_results(results)


@hydra.main(config_path="config", config_name="pipeline_config.yaml", version_base=None)
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    pipeline = Pipeline(cfg)


if __name__ == "__main__":
    main()
