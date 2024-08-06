import torch
from PIL import Image
from prob.models import build_model
from prob.config import get_cfg

import torchvision.transforms as T


def load_checkpoint(checkpoint_path):
    '''
    TODO:needs work, load the correct model, conf?
    '''
    cfg = get_cfg()
    cfg.merge_from_file(checkpoint_path)  # Load the checkpoint config
    model = build_model(cfg)  # Build the model using the config
    model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))  # Load the model weights
    return model


def preprocess_image(images:list):
    '''
    Currently not needed nor operational
    Replace transformations with different transforms? or delete?
    '''
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transformed = transform(image).unsqueeze(0)
    return None

def run_inference(model,image:tensor,preprocessing_fn:callable):
    ''' 
    Common inference function for all models
    '''
    model.eval()
    image = preprocess_image(image,preprocessing_fn)
    with torch.no_grad():
        output = model(image)
    return output

def detect_object(image, model):
    model = load_checkpoint(checkpoint_path)
    output = run_inference(model, image_path)
    # Process the output to get the detected objects
    # ...
    return detected_objects

def detect_objects(model_path:str, images:list):
    model = load_checkpoint(model_path)
    for image in images:
        bboxes = run_inference(model)
    return detected_objects