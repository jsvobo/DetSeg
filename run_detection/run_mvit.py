from mvits_for_class_agnostic_od.inference.main import  run_inference
from mvits_for_class_agnostic_od.models.model import Model
import os

import sys

# Assuming the script is run from the root of your main project
sub_project_path = os.path.join(os.path.dirname(__file__), 'mvits_for_class_agnostic_od')
sys.path.append(sub_project_path)

''' 
TODO: make a loader for the data, independent of the model???
bit too hard, maybe just a parent class
'''

def detect_objects(dataset='COCO'):
    ''' 
    code loosely interpreted from https://github.com/mmaaz60/mvits_for_class_agnostic_od/ inference/main.py , function main()
    this function prepares datasets and model for inference and runs the inference
    results are saved in the output directory 
    TODO: configs
    '''

    #hard coded for now, need to change to conf
    output_path =  '/mnt/vrg2/imdec/out/COCO/mvit'
    images_dir =  './Datasets/COCO/train2017'
    text_query= 'all objects' #default query? 

    multi_crop = False #only on DOTA dataset for some reason
    model_name = "mdef_detr_minus_language" #model name from ['mdef_detr','mdef_detr_minus_language']
    checkpoints_path = '/mnt/vrg2/imdec/models/detectors/mdef_detr_minus_language.pth' #path to model checkpoint
    model = Model(model_name, checkpoints_path).get_model()

    run_inference(model, images_dir, output_path, caption=text_query, multi_crop=multi_crop) #TODO: try run?
    

def load_mvit_results(dataset='COCO'):
    '''
    TODO: write this function!!
    '''
    #parse location from config
    path_out = '/mnt/vrg2/imdec/out'
    model_name= 'mvit'
    
    #load bboxes and return them

    #load other information??
    return bboxes


if __name__ == "__main__": 
    detect_objects()
    print("Done")
    #bboxes = load_mvit_results()
    #print("size:" + str(len(bboxes)))