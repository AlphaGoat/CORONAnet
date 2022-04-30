import math


def log2(x):
    return math.log2(x)


def get_downsampled_image_dims(original_image_dims, downsample_factor=1):
    """
    Get dimensions for downsampled image, to nearest power of 2

    :params original_image_dims: tuple or list of image dims (height, width, channel)
    :params downsample_factor: factor to downsample each dimension of input image
    """
    image_height = original_image_dims[0]
    image_width = original_image_dims[1]

    reduce_height = image_height // downsample_factor
    reduce_width = image_height // downsample_factor

    downsample_height = 2 ** int(log2(reduce_height))
    downsample_width = 2 ** int(log2(reduce_width))

    if len(original_image_dims) == 3:
        downsample_dims = (downsample_height, downsample_width, original_image_dims[2])
    else:
        downsample_dims = (downsample_height, downsample_width)

    return downsample_dims
