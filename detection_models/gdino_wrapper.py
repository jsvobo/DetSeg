import torch
import utils
import numpy as np

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from detection_models.dummy_detectors import BaseDetectorWrapper
from omegaconf import DictConfig, OmegaConf


class GrDINO(BaseDetectorWrapper):

    def compose_text_prompt(self, all_classes):
        self.text_prompt = ".".join(all_classes)
        self.all_classes = all_classes.copy()
        self.all_classes.append("")

    def classes_to_indices(self, list_of_classes):
        return [self.all_classes.index(c) for c in list_of_classes]

    def indices_to_classes(self, list_of_indices):
        return [self.all_classes[i] for i in list_of_indices]


class GroundingDinoTiny(GrDINO):

    def __init__(self, device="cuda", cfg: DictConfig = None, all_classes=None):
        self.model_id = "IDEA-Research/grounding-dino-tiny"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)

        # prepare prompt for grounding dino
        self.compose_text_prompt(all_classes)

        if (
            cfg is None
        ):  # if not in a pipeline, but in a notebook, load a "manual" config from a specific loaction
            cfg = OmegaConf.load("./config/detector/gdino_manual.yaml")
        self.box_threshold = cfg["box_threshold"]
        self.text_threshold = cfg["text_threshold"]

    def detect_individual(self, image):
        # image is a single image, returns detected boxes
        text = self.text_prompt
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_size = image.shape[:2]  # image.size is (width, height)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[target_size],
        )
        results = results[0]
        results["boxes"] = torch.round(results["boxes"].cpu()).type(torch.int32)
        return results

    def detect_boxes(self, items):
        # takes list of item, as returned by dataset loader(s)
        results = []
        for item in items:
            image = item["image"]
            output = self.detect_individual(image)
            results.append(
                {"boxes": output["boxes"], "labels": [0 for _ in output["labels"]]}
            )
        return results

    def detect_batch(self, images, metadata=None):
        """
        images: list of images
        right now sequential, but grounding dino has some batching options for the future.
        return format same is the pipeline needs to work
        detected boxes,attention points for each box, labels of those points, class detected for each box
        """
        detected_boxes = []
        detected_classes = []
        for image in images:
            output = self.detect_individual(image)
            detected_boxes.append(output["boxes"])  # just boxes
            detected_classes.append([0 for _ in output["labels"]])  # no classes here
        return (
            detected_boxes,
            None,  # no attention points
            None,  # no point labels
            detected_classes,
        )  # return outputs for all images


if __name__ == "__main__":
    from datasets.dataset_loading import CocoLoader, get_coco_split

    transforms = None
    coco_train_dataset = CocoLoader(get_coco_split(split="val"), transform=transforms)
    all_classes = coco_train_dataset.get_classes()

    gd = GroundingDinoTiny(device="cuda", cfg=None, all_classes=all_classes)

    batch = [item["image"] for item in coco_train_dataset.get_amount(5)]
    outputs = gd.detect_batch(batch)
    print(outputs)
