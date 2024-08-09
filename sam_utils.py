import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image


def show_mask(mask, ax, random_color=False):
    '''
    Interpreted from https://github.com/facebookresearch/segment-anything
    '''
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=300):
    '''
    Interpreted from https://github.com/facebookresearch/segment-anything
    '''
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    

def print_masks_point(masks,scores,img):
    '''
    Interpreted from https://github.com/facebookresearch/segment-anything
    '''
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show() 

def plot_box(box, ax, color='red', linewidth=2):
    '''
    bounding boxes in format x0,y0,x1,y1 (main diagonal points)
    Plots the boc on the ax from plt
    '''
    ax.plot([box[0], box[2], box[2], box[0], box[0]],
            [box[1], box[1], box[3], box[3], box[1]],
            color=color, linewidth=linewidth)

def print_masks_boxes(masks, boxes, img):
    '''
    bounding boxes in format x0,y0,x1,y1 (main diagonal points)
    rints all masks and boxes on the image
    '''
    scale=8
    opacity=0.8
    box_width=2
    BG_MASK=False
    
    assert(len(masks)==len(boxes))

    plt.figure(figsize=(scale, scale))
    plt.imshow(img) #first image
    plt.axis('off')

    cmap =colormaps['viridis']
    alpha = np.ones_like(img)[:,:,0]*opacity

    mask_sum = np.zeros_like(img)[:,:,0]
    for i,mask in enumerate(masks):
        mask_sum = np.maximum(mask_sum,mask*(i+1)) #layer masks
    if not BG_MASK: alpha[np.where(mask_sum==0)] = 0
    else: alpha[np.where(mask_sum==0)] = opacity/2
    plt.imshow(mask_sum, cmap=cmap, alpha=alpha) #TODO:  show just the masked part with color and not the rest?

    num_boxes=len(boxes)
    for i,box in enumerate(boxes): #all masks
        plot_box(box,plt.gca(),linewidth=box_width) #color=cmap(i/num_boxes),
    plt.show()


def prepare_sam(device):
    '''
    Load SAM-1 predictor and model itself (needed for some functions) 
    '''
    from segment_anything import SamPredictor, sam_model_registry
    sam_checkpoint = "/datagrid/personal/janoukl1/out/ImDec/ckpts/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return SamPredictor(sam),sam

def select_best_mask(masks, scores):
    idx = np.argmax(scores)
    return masks[idx], scores[idx]



def crop_xywh(img,mask,box,x0,y0,w,h):
    ''' 
    Input: 
        PIL image, OR ndarray image,
        np.array mask, 
        box in format x0,y0,x1,y1, 
        x0,y0 is the left upper corner of the crop
        w,h are the cropped window sizes
    
    Description:
    Crop the image mask and bounding box, starting at coords x0,y0 at the left upper corner
    w,h sets the size of the resulting window
    '''
    x0,y0,w,h = np.int32(x0),np.int32(y0),np.int32(w),np.int32(h) #to int

    if img.__class__.__name__ =="Image": #PIL image
        cropped_img = img.crop((x0,y0,x0+w,y0+h))
    elif img.__class__.__name__ =="ndarray": #numpy array
        cropped_img = img[y0:y0+h,x0:x0+w]
    else:
        raise ValueError("Unknown image type")
    
    box_coords = [box[0]-x0,box[1]-y0,box[2]-x0,box[3]-y0] #subtract corner from the box
    cropped_mask = mask[y0:y0+h,x0:x0+w] #need to also crop the w,h

    return cropped_img,cropped_mask,box_coords

def crop_xyxy(img,mask,box,x0,y0,x1,y1):
    ''' 
    Input: 
        PIL image, OR ndarray image,
        np.array mask, 
        box in format x0,y0,x1,y1, 
        x0,y0 is the left upper corner of the crop
        x1,y1 is the left upper corner of the crop
    
    Description:
    Crop the image mask and bounding box, starting at coords x0,y0 at the left upper corner
    w,h sets the size of the resulting window
    '''
    x0,y0,x1,y1 = np.int32(x0),np.int32(y0),np.int32(x1),np.int32(y1) #to int

    if img.__class__.__name__ =="Image": #PIL image
        cropped_img = img.crop((x0,y0,x1,y1))
    elif img.__class__.__name__ =="ndarray": #numpy array
        cropped_img = img[y0:y1,x0:x1]
    else:
        raise ValueError("Unknown image type")
    
    box_coords = [box[0]-x0,box[1]-y0,box[2]-x0,box[3]-y0] #subtract corner from the box
    cropped_mask = mask[y0:y1,x0:x1] #need to also crop the w,h

    return cropped_img,cropped_mask,box_coords