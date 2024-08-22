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
        Infers masks for the given items.
        Args:
            items (list): A list of items for which masks need to be inferred.
            boxes (list, optional): A list of bounding boxes corresponding to the items. Defaults to None.
            points (list, optional): A list of points corresponding to the items. Defaults to None.
            point_labels (list, optional): A list of labels for the points. Defaults to None.
        Returns:
            List of dicts containing masks and their scores

        Sequentially processes the images, returns list of dicts
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
