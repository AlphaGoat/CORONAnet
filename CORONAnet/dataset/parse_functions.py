import tensorflow as tf
from functools import partial
from typing import List, Tuple, Dict
from CORONAnet.math import apply_transform


def single_frame_parse_function(example_proto, return_filename=False):
    """
    Parse example proto from tfrecord 
    """
    # feature to extract from example_proto
    features = {
            "height": tf.io.FixedLenFeature([], dtype=tf.int64),
            "width": tf.io.FixedLenFeature([], dtype=tf.int64),
            "channels": tf.io.FixedLenFeature([], dtype=tf.int64),
            "label": tf.io.FixedLenFeature([], dtype=tf.int64),
            "image_raw": tf.io.FixedLenFeature([], dtype=tf.string),
            "filename": tf.io.VarLenFeatures(tf.string)
    }

    # Parse the example 
    parsed_features = tf.io.parse_single_example(serialized=example_proto, 
            features=features)
    width = tf.cast(features['width'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    num_channels = tf.cast(features['channels'], tf.int32)

    # Construct image tensor
    image_raw = tf.sparse.to_dense(parsed_features['image_raw'])
    image = tf.io.decode_raw(image_raw, channels=num_channels)
    image = tf.reshape(image, [height, width, num_channels])

    label = parsed_features['label']

    filename = parsed_features['filename']

    if return_filename:
        return image, label, filename 
    else:
        return image, label


def multi_frame_parse_function(example_proto: tf.train.Example, 
                               return_filename: bool=False, 
                               resize_dims: Tuple[int] or List[int]=None,
                               target_labels: str or Tuple[str] or List[str] or Dict[str, tf.dtype]=[], 
                               target_transforms: str or Dict[str, str] or List[str]=None):
    """
    Parse multi-frame example proto from tfrecord 

    :param example_proto: TensorFlow example proto buffer for instance 
    :param return_filename: Flag to set to return filename with parsed proto features 
    :param resize_dims: Tuple of dimensions to resize serialized image to (height, width)
    :param target_labels: Either a string of a single target label name if only one target is 
     going to be used, a list/tuple of target label names, or a dictionary whose keys are the 
     target label names and whose values are the TensorFlow datatype to parse the feature as
    :param target_transforms:  transforms to apply to labels values after they are parsed from 
     the protobuf. Can be a string if the same transform is going to  be applied to all labels, 
     a list of transforms to apply to labels in order, or a dictionary whose keys are the target 
     label name and whose values are the transforms to apply to that label

    :return image: tf.float32 input image
    :return target_tensor: tf.float32 tensor of target values
    """
    # features to extract from example_proto
    features = {
            "height": tf.io.FixedLenFeature((), dtype=tf.int64),
            "width": tf.io.FixedLenFeature((), dtype=tf.int64),
            "channels": tf.io.FixedLenFeature((), dtype=tf.int64),
            "class_id": tf.io.FixedLenFeature((), dtype=tf.int64),
            "num_frames": tf.io.FixedLenFeature((), dtype=tf.int64),
            "sequence_raw": tf.io.FixedLenFeature((), dtype=tf.string),
    }

    if isinstance(target_labels, dict):
        for label in target_labels.keys():
            features[label] = tf.io.FixedLenFeature((), dtype=target_labels[label])
    elif isinstance(target_labels, list) or isinstance(target_labels, tuple):
        for label in target_labels:
            features[label] = tf.io.FixedLenFeature((), dtype=tf.float32)
    elif isinstance(target_labels, str):
        features[target_labels] = tf.io.FixedLenFeature((), dtype=tf.float32)
        target_labels = [target_labels]
    else:
        raise ValueError(f"target_labels an unrecognized type {type(target_labels)}.")

    # Parse the example 
    parsed_features = tf.io.parse_single_example(serialized=example_proto, 
            features=features)
    width = tf.cast(parsed_features['width'], tf.int32)
    height = tf.cast(parsed_features['height'], tf.int32)
    class_id = tf.cast(parsed_features['class_id'], tf.int32)
    num_channels = tf.cast(parsed_features['channels'], tf.int32)
    num_frames = tf.cast(parsed_features['num_frames'], tf.int32)

    # Construct multiframe image tensor
    image_raw = parsed_features['sequence_raw']
    image_sequence = tf.io.decode_raw(image_raw, tf.uint8)
    image_sequence = tf.reshape(image_sequence, [num_frames, height, width, num_channels])
    image_sequence = tf.cast(image_sequence, tf.float32)

    if resize_dims is not None:
        image_sequence = tf.map_fn(
            lambda image: tf.image.resize(image, (resize_dims[0], resize_dims[1])),
            image_sequence,
        )

    def _transform_wrapper(val, label, i=0):
        if target_transforms is None:
            return val
        elif isinstance(target_transforms, dict):
            return apply_transform(val, target_transforms[label])
        elif isinstance(target_transforms, list) or isinstance(target_transforms, tuple):
            return apply_transform(val, target_transforms[i])
        elif isinstance(target_transforms, str):
            return apply_transform(val, target_transforms)
        else:
            raise ValueError(f"label_transform is an unrecognized type {type(target_transforms)}")

    label_values = list()
    if isinstance(target_labels, dict):
        for i, label in enumerate(target_labels.keys()):
            label_val = tf.cast(parsed_features[label], tf.float32)
            label_val = _transform_wrapper(label_val, label, i)
            label_values.append(label_val)
    else:
        for i, label in enumerate(target_labels):
            label_val = tf.cast(parsed_features[label], tf.float32)
            label_val = _transform_wrapper(label_val, label, i)
            label_values.append(label_val)

    target_tensor = tf.stack(label_values, axis=-1)

    return image_sequence, target_tensor


def get_parse_function(parse_descriptor):

    if parse_descriptor.lower() == "multi_frame_parse":
        return multi_frame_parse_function
    elif parse_descriptor.lower() == 'single_frame_parse':
        return single_frame_parse_function
    else:
        raise NotImplementedError(f"{parse_descriptor} not implemented.")
