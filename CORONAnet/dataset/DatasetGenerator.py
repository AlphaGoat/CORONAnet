"""
Dataset generator for CORONAnet. Uses serialized tfrecord format 

Author: Peter Thomas 
Date: 17 April 2022
"""
import os
import glob
import numpy as np
import tensorflow as tf
from functools import partial
from typing import Callable, Tuple, List, Union, Optional

from CORONAnet.math import apply_transform


class DatasetGenerator():
    def __init__(self,
                 tfrecord_path: str,
                 parse_function: Callable,
                 image_shape: Union[Tuple[int, int, int], List[int]],
                 batch_size: int=32,
                 buffer_size: int=32,
                 max_len_sequence: int=20,
                 return_filename: bool=False,
                 oversampling_technique: str=None,
                 target_labels: List[str]=["peak_intensity"],
                 target_transforms: Union[str, List[str]]="log-transform",
                 intensity_bins: List[float]=[],
                 oversampled_distribution: List[float]=[0.3, 0.7],
                 initial_distribution: List[float]=[0.02, 0.98],
                 repeat: Optional[int]=None,
                 ):
        """
        Initialize data generator for coronagraph data 

        :param tfrecord_path: path to tfrecords file or folder containing tfrecords  
        :param parse_function: function to use to parse tfrecords serialized examples 
        :param batch_size: batch size to use for training 
        :param buffer_size: Number of example to load into buffer in memory 
        :param return_filename: flag to set to return filename along with image and labels 
        :param oversampling_technique: technique to use to oversample example in data, if 
                                       desired 
        :param intensity_bins: list of intensity thresholds to use to divide different 
                               "classes" of data (length needs to be n+1, where 'n' is
                               the number of classes)
        :param oversampled_distribution: Distribution of oversampled data. Needs to be 
                                         length 'n', where 'n' is the number of classes
        :param initial_distribution: Initial distribution of data (if not provided, we'll 
                                     iterate over the dataset once to get the distribution).
                                     Is length 'n', where 'n' is the number of classes
        """
        self.tfrecord_path = []
        if type(tfrecord_path) == list:
            self.tfrecord_path = tfrecord_path
        elif os.path.isdir(tfrecord_path):
            for filepath in glob.glob(os.path.join(tfrecord_path, "*.tfrecord")):
                self.tfrecord_path.append(filepath)
        else:
            self.tfrecord_path.append(tfrecord_path)

        self.image_shape = image_shape
        self.batch_size = batch_size
        self.return_filename = return_filename
        self.repeat = repeat

        # prepare proto parse function
        self.target_labels = target_labels
        self.parse_function = partial(parse_function, 
                                      resize_dims=image_shape,
                                      max_len_sequence=max_len_sequence,
                                      target_labels=target_labels)

        # target transform to apply to each label
        self.target_transforms = target_transforms

        # number of instances to load into memory
        self.buffer_size = buffer_size

        # Oversampling information for SEP and high-speed / large width 
        # non-SEP instances
        self.oversampling_technique = oversampling_technique 

        # if intensity bins are provided, use those to distinguish 
        # different classes. Else, just use threshold value as cutoff
        if intensity_bins:
            self.intensity_bins = intensity_bins
        else:
            self.intensity_bins = [-np.inf, 10.0, np.inf]

        # make sure that the distributions provided are compatible with 
        # the intensity bins provided
        self.oversampled_distribution = oversampled_distribution
        self.initial_distribution = initial_distribution

        # if some oversampling technique is provided, ensure that the provided 
        # are consistent
        if self.oversampling_technique is not None:
            if len(self.oversampled_distribution) != len(self.intensity_bins) - 1:
                raise ValueError(f"Number of thresholds in intensity bins does not correspond    "
                                 + f"with oversampled distribution ({len(self.intensity_bins)}) provided, "
                                 + f"while oversampled distribution implies ({len(self.oversampled_distribution)} "
                                 f"bins are required")

            if len(self.initial_distribution) != len(self.intensity_bins) - 1:
                raise ValueError(f"Number of thresholds in intensity bins does not correspond    "
                                 + f"with initial distribution ({len(self.intensity_bins)}) provided, "
                                 + f"while initial distribution implies ({len(self.oversampled_distribution)} "
                                 f"bins are required")

        # construct dataset pipeline
        self.dataset = self.build_pipeline()


    def build_pipeline(self):

        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset.shuffle(self.buffer_size)
        
        # parse example proto
        dataset = dataset.map(self.parse_function)

        # normalize images 
        dataset = dataset.map(image_normalization)

        # get the initial distribution, if one was not provided
        if self.initial_distribution is None:
            self.initial_distribution = compute_distribution(dataset, self.intensity_bins)

        # apply oversampling, if desired
        dataset = self.apply_oversampling(dataset)

        # apply target transforms (I don't do this earlier, as oversampling 
        # bins rely on original peak intensity values, before transforms are applied)
        dataset = dataset.map(self.apply_target_transforms)

        # batch dataset
        dataset = dataset.batch(self.batch_size)

        if self.repeat is not None:
            dataset = dataset.repeat(self.repeat)

        # return reference to data pipeline
        return dataset


    def apply_target_transforms(self, image, labels):
        """
        Apply transforms to target labels, if desired 
        """
        if self.target_transforms is None:
            return image, labels

        elif isinstance(self.target_transforms, str):
            transformed_labels = apply_transform(labels, self.target_transforms)
            return image, transformed_labels

        elif isinstance(self.target_transforms, list) or isinstance(self.target_transforms, tuple):
            targets = tf.unstack(labels, axis=-1)
            transformed_targets = list()
            for target, transform in zip(targets, self.target_transforms):
                target = apply_transform(target, transform)
                transformed_targets.append(target)

            transformed_labels = tf.stack(transformed_targets, axis=-1)
            return image, transformed_labels

        elif isinstance(self.target_transforms, dict):
            targets = tf.unstack(targets, axis=-1)
            transformed_targets = list()
            for target, label_name in zip(targets, self.target_labels):
                transform = self.target_transforms[label_name]
                target = apply_transform(target, transform)
                transformed_targets.append(target)

            transformed_labels = tf.stack(transformed_targets, axis=-1)
            return image, transformed_labels

        else:
            raise ValueError("target_transforms an unrecognized type " +
                             f"({type(self.target_transforms)}")


    def apply_oversampling(self, dataset):
        if self.oversampling_technique is None:
            return dataset
        elif self.oversampling_technique == 'random-oversampling':
            dataset =  dataset.apply(
                tf.data.experimental.rejection_resample(
                class_func=lambda x1, x2: self.map_intensity_to_class(x1, x2),
                target_dist=self.oversampled_distribution,
                initial_dist=self.initial_distribution,
                seed=42)
            )

            # for some reason, the class index is returned with the image and labels 
            # after applying this function. We don't want that, so this is a stupid wrapper
            # to remove the first return val after the rejection_resampler is applied
            def _toss_first_val(first_val, args):
                return args[0], args[1]

            return dataset.map(_toss_first_val)

        else:
            raise NotImplementedError(f"Oversampling technique {self.oversampling_technique} not implemented.")


    def map_intensity_to_class(self, image, labels):
        """
        Map target intensity value for an example to a class for 
        oversampling purposes.
        """
        try:
            # Get peak intensities and see which bin edges the intensity value is 
            # greater than. The bin to assign this instance will be the sum of the 
            # bin edges that the label is greater than minus 1
            peak_intensity_idx = self.target_labels.index('peak_intensity')
            peak_intensities = labels[peak_intensity_idx] 
            less = tf.math.greater_equal(peak_intensities, self.intensity_bins)
            class_indices = tf.math.reduce_sum(tf.cast(less, tf.int32)) - 1
        except IndexError:
            # if peak intensity is not one of the target labels, then we don't have 
            # a way to divide instances into classes (we might encode some of the recorded 
            # features of events later, such as speed, half-angle, etc., to allow for 
            # class divisions on other feature values. In the mean time, just return the 
            # same class of all instances)
            return tf.constant(0, tf.int32)

        return class_indices


def image_normalization(image, label): 
    image = tf.image.per_image_standardization(image)
    return image, label


def compute_distribution(dataset, intensity_bins):
    """
    Compute the distribution of classes in a dataset based on a set of 
    intensity bin edges 
    """
    num_instances = tf.Variable((0), dtype=tf.int32)
    instances_per_bin = np.zeros(len(intensity_bins) - 1)
    iterator = dataset.make_one_shot_iterator()

    _, next_label = iterator.get_next()

    greq_cond = tf.math.greater_equal(next_label[..., 0], intensity_bins[tf.newaxis, :])
    class_indices = tf.math.reduce_sum(tf.cast(greq_cond, tf.int32), axis=-1) - 1
    instances_per_bin = tf.gather(instances_per_bin, class_indices) + 1 
    update = num_instances.assign_add(1)

    with tf.Session() as sess:
        try:
            while True:
                _, distribution = sess.run([update, instances_per_bin])
        except tf.errors.OutOfRangeError:
            pass

    normalized_distribution = instances_per_bin / num_instances

    return normalized_distribution
