import torch
from PIL import Image
from prob.models import build_model
from prob.config import get_cfg

import torchvision.transforms as T


def load_model(checkpoint_path:str, model:str='prob'):
    '''
    Load the specified model from the checkpoint path
    '''
    if model == "prob":
        model = prob_load_checkpoint(checkpoint_path)
    else:
        raise ValueError("Invalid model name")
    return model


def prob_load_checkpoint(checkpoint_path:str,cfg:dict):
    '''
    TODO: needs work, load the correct model, conf?
    '''
    cfg = get_cfg()
    cfg.merge_from_file(checkpoint_path)  # Load the checkpoint config
    model = build_model(cfg)  # Build the model using the config
    model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS))  # Load the model weights
    return model


def preprocess_image(image):
    '''
    Currently not needed nor operational
    Replace transformations with different transforms? or delete?
    '''
    transform = T.Compose([ 
        T.Resize((800, 800)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transformed = transform(image).unsqueeze(0)
    return None

def run_inference_one(model,image_tensor:tensor,transforms:list):
    ''' 
    Common inference function for all models
    '''
    model.eval()
    image_tensor_processes = preprocess_image(image_tensor,transforms)
    with torch.no_grad():
        output = model(image)
    return output


def detect_objects(model,images:list):
    predicted_bboxes = []
    for image in images:
        bboxes = run_inference(model)
        predicted_bboxes.append(bboxes)
    return predicted_bboxes


if __name__ == "__main__": 
    #TODO: minimal running example, load 1 image for test?
    model="prob"
    model_path = "path/to/model" #TODO:fill

    model = load_model(model_path,model)
    images=[] #mock list of images here
    predicted_bboxes = detect_objects(model,images)
    #visualize into a file???? (no code yet)