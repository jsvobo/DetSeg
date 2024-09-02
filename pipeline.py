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

        dataset_class = cfg.class_name
        get_path = cfg.split_fn

        path = getattr(datasets, get_path)(split=split, year=year, root=root)
        dataset = getattr(datasets, dataset_class)(path, transform=None)
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
        cfg.dataset.year = str(cfg.dataset.year)
        return cfg

    def print_results(self, result_dict):
        """
        Structured output of the results
        """
        mean_seg_IoU = result_dict["mean seg IoU"]
        weighed_seg_IoU = result_dict["weighted seg IoU"]
        mean_det_IoU = result_dict["mean det IoU"]
        weighed_det_IoU = result_dict["weighted det IoU"]

        print("\nResults:")
        print("Detection metrics:")
        print(f"    Mean IoU: {round(float(mean_det_IoU),3)}")
        print(f"    Weighted IoU: {round(float(weighed_det_IoU),3)}")
        print("Segmentation metrics:")
        print(f"    Mean IoU: {round(float(mean_seg_IoU),3)}")
        print(f"    WeightedIoU: {round(float(weighed_seg_IoU),3)}")

        print("\nClassless mAP:")
        print("Detection metrics:")
        pprint.pprint(result_dict["classless mAP - detection"])  # add more dicts here?
        if self.cfg.segmentation.class_name != "None":
            print("Segmentation metrics:")
            pprint.pprint(result_dict["classless mAP - segmentation"])

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

        # Instantiate the Evaluator with the requested parameters
        evaluator = Evaluator(
            cfg=cfg.evaluator,
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
        result_dict, array_masks, array_boxes, index_array = evaluator.get_metrics()

        if print_results:
            self.print_results(result_dict)

        # save result dictionary and metadata to a file.
        if save_results:
            utils.save_results(
                result_dict=result_dict,
                array_masks=array_masks,  # array of IoU for masks and boxes
                array_boxes=array_boxes,
                index_array=index_array,
                cfg=cfg,
            )


@hydra.main(config_path="config", config_name="pipeline_config.yaml", version_base=None)
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    pipeline = Pipeline(cfg)


if __name__ == "__main__":
    main()
