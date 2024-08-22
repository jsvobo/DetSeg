class BaseSegmentWrapper:
    """
    Parent class for sam based segmentation wrappers.
    """

    def __init__(self, device, model):
        """
        Initializes the BaseSegWrapper class.
        Args:
            device (str): The device to be used for computation.
            model: The segmentation model size.
        Returns:
            None
        """
        pass

    def infer_masks(self, items, boxes=None, points=None, point_labels=None):
        """
        Does not use batching, sequentially processes the images, returns list of dicts
            Each dict has masks and their scores for one image

        Args:
            items (list): List of dicts with images and GT annotations
            boxes (list): List of bounding boxes for prompting, Optional, then use GT boxes
            points (list, optional): List of point coordinates for prompting. Defaults to None.
            point_labels (list, optional): List of point labels for prompting. Defaults to None.

        Returns:
            dict: A dictionary containing the inferred masks and their scores.
                - "masks" (list): List of inferred masks.
                - "scores" (list): List of mask scores.
        """
        pass

    def infer_batch(
        self,
        images,
        boxes=None,
        point_coords=None,
        point_labels=None,
    ):
        """
        Perform inference on a batch of images.
        Args:
            images (Tensor): A tensor containing the input images.
            boxes (Tensor, optional): A tensor containing the bounding boxes for each image. Defaults to None.
            point_coords (Tensor, optional): A tensor containing the coordinates of points of interest for each image. Defaults to None.
            point_labels (Tensor, optional): A tensor containing the labels of points of interest for each image. Defaults to None.
        Returns:
            Tensor: A tensor containing the predicted outputs for each box
        """
        pass

    def get_sam(self):
        """
        Returns the sam object.
        """
        return self.sam

    def get_image_size(self):
        """
        Returns the image size needed inside sam
        (for batching, where you resize the images yourself)
        """
        return self.sam.image_encoder.img_size
