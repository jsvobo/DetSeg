import torch
import utils
import numpy as np

from omegaconf import DictConfig, OmegaConf

from detection_models.gdino_wrapper import GrDINO
from groundingdino.util.inference import Model, predict
from torchvision.transforms.functional import to_pil_image


class GroundingDinoFull(Model, GrDINO):
    # extends Model from groundingdino/utils/inference.py

    def __init__(self, device="cuda", cfg: DictConfig = None, all_classes=None):
        super().__init__(
            model_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path="GroundingDINO/weights/groundingdino_swint_ogc.pth",
            device=device,
        )

        # prepare prompt for grounding dino
        self.classes = all_classes
        self.compose_text_prompt(self.classes)  # function of GrDINO

        if (
            cfg is None
        ):  # if not in a pipeline, but in a notebook, load a "manual" config from a specific loaction
            print("No config provided to grounding dino, loading default")
            cfg = OmegaConf.load("./config/detector/gdino_ntbk.yaml")

        self.box_threshold = cfg["box_threshold"]
        self.text_threshold = cfg["text_threshold"]

    def detect_individual(self, image):
        processed_image = self.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(  # use the built in predict function
            model=self.model,
            image=processed_image,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        phrases = [
            phrase.replace("#", "") for phrase in phrases
        ]  # clear some of the tokens (like '##bling lizard')

        confidences = logits.cpu().detach()
        class_id = self.phrases2classes(
            phrases=phrases, classes=self.classes
        )  # function of Model, propose some class indices based on output phrases

        class_id = [-1 if id is None else id for id in class_id]
        print(class_id)
        sizes = image.shape[:2]  # (h,w) from numpy array img, to scale back up
        boxes = boxes * torch.tensor(  # scale back up
            [sizes[1], sizes[0], sizes[1], sizes[0]],
            dtype=torch.float32,  # * (w,h,w,h)' @ boxes
        )
        boxes = utils.box_torch2xyxy(
            boxes
        )  # transform as matrix from x,y,w,h to x0,y0,x1,y1

        results = {
            "boxes": torch.round(boxes.cpu().detach()).type(torch.int32),
            "labels": class_id,  # list of lists. for every box, more than 1 class suggestion is possible
            "scores": confidences,
        }
        return results

    def detect_boxes(self, items):
        # takes list of item, as returned by dataset loader(s)
        results = []
        for item in items:
            image = item["image"]
            output = self.detect_individual(image)
            results.append(output)
        return results

    def detect_batch(self, images, metadata=None):
        """
        images: list of images
        return format same is the pipeline needs to work
        detected boxes,attention points for each box, labels of those points, class detected for each box
        """
        boxes, classes, scores = [], [], []
        for image in images:
            output = self.detect_individual(image)
            boxes.append(output["boxes"])  # just boxes
            classes.append(
                [0 for _ in output["labels"]]
            )  # no classes here, labels are text, not numbers!
            scores.append(output["scores"])  # confidence scores
        return {
            "boxes": boxes,
            "class_labels": classes,
            "confidence": scores,
            "attention_points": None,  # no attention points
            "point_labels": None,  # no point labels
        }  # return outputs for all images


def test_grounding_dino_full():
    from datasets.coco_wrapper import CocoLoader, get_coco_split

    transforms = None
    coco_train_dataset = CocoLoader(get_coco_split(split="val"), transform=transforms)
    all_classes = coco_train_dataset.get_classes()

    gd = GroundingDinoTiny(device="cuda", cfg=None, all_classes=all_classes)

    batch = [item["image"] for item in coco_train_dataset.get_amount(5)]
    outputs = gd.detect_batch(batch)
    print(outputs["class_labels"])


if __name__ == "__main__":
    test_grounding_dino_full()
