import torch
import utils
import numpy as np

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from detection_models.dummy_detectors import BaseDetectorWrapper
from omegaconf import DictConfig, OmegaConf


class GrDINO(BaseDetectorWrapper):

    def compose_text_prompt(self):
        self.got_classes = False

        if self.all_classes is not None:
            self.got_classes = True
            self.text_prompt = ". ".join(self.all_classes)
            self.all_classes = self.all_classes.copy()
            self.all_classes.append("")

            print(self.all_classes)
            print(self.text_prompt)
        else:
            self.text_prompt = "an item. an object. hidden object. entity. a thing. stuff. small object. large object. hidden object"

    def classes_to_indices(self, list_of_classes):
        if not self.got_classes:
            return [0 for c in list_of_classes]
        return [self.all_classes.index(c) for c in list_of_classes]

    def indices_to_classes(self, list_of_indices):
        if not self.got_classes:
            return ["" for i in list_of_indices]
        return [self.all_classes[i] for i in list_of_indices]


class GroundingDinoTiny(GrDINO):

    def __init__(self, device="cuda", cfg: DictConfig = None, all_classes=None):
        self.model_id = "IDEA-Research/grounding-dino-tiny"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)

        self.all_classes = all_classes
        self.compose_text_prompt()

        if cfg is None:
            cfg = OmegaConf.load("./config/detector/gdino_tiny.yaml")
        self.box_threshold = cfg["box_threshold"]
        self.text_threshold = cfg["text_threshold"]

    def detect_individual(self, image):
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
        results["boxes"] = torch.round(results["boxes"].cpu()).type(torch.int16)
        return results

    def detect_batch(self, images, metadata=None):
        """
        images: list of images
        """
        detected_boxes = []
        detected_classes = []
        detected_class_names = []
        for image in images:
            output = self.detect_individual(image)
            detected_boxes.append(output["boxes"])
            detected_classes.append(self.classes_to_indices(output["labels"]))
        return (
            detected_boxes,
            None,  # no attention points
            None,  # no point labels
            detected_classes,
        )  # return outputs for all images

    def detect_boxes(self, items):
        images = [item["image"] for item in items]
        outputs = self.detect_batch(images)
        results = []
        for i in range(len(items)):
            results.append(
                {
                    "boxes": outputs[0][i],
                    "labels": outputs[3][i],
                }
            )
        return results


if __name__ == "__main__":
    from datasets.dataset_loading import CocoLoader, get_coco_split

    transforms = None
    coco_train_dataset = CocoLoader(get_coco_split(split="val"), transform=transforms)
    all_classes = coco_train_dataset.get_classes()

    gd = GroundingDinoTiny(device="cuda", cfg=None, all_classes=all_classes)

    batch = [item["image"] for item in coco_train_dataset.get_amount(5)]
    outputs = gd.detect_batch(batch)
    print(outputs)
