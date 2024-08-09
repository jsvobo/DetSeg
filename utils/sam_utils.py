import numpy as np
from segment_anything import SamPredictor, sam_model_registry


def prepare_sam(device):
    '''
    Load SAM-1 predictor and model itself (needed for some functions) 
    '''
    sam_checkpoint = "/datagrid/personal/janoukl1/out/ImDec/ckpts/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return SamPredictor(sam),sam

def select_best_mask(masks, scores):
    idx = np.argmax(scores)
    return masks[idx], scores[idx]


