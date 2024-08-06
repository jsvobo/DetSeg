
''' 

'''

class AnnotationLoader():
    ''' 
    Class which loads the annotations from a file.
    Obtained Bounding Boxes, segmentation masks etc.
    Load single image annotation at once 
    load metadata (time, experiment description, classes, metrics etc.)
    '''
    pass


class AnnotationSaver():
    ''' 
    Class which saves the annotations to a file.
    Obtained Bounding Boxes, segmentation masks etc.
    precise format of the file is not yet decided
    .json files (for each image one + metadata_annotation +all_in_one)
    '''
    pass

class image_saver():
    ''' 
    Class which saves the images with bboxes drawn on them.
    For visualization purposes.
    '''
    pass

