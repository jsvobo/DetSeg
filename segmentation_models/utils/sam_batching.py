def _prepare_image_for_batch(device, image, resize_transform):
    """
    Prepare the image for batch processing
    """
    image = resize_transform.apply_image(image)  # wants HWC (numpy) not CHW (torch)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()  # CHW
