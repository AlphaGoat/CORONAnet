"""
Module for implementing loss functions for CORONAnet 

Author: Peter Thomas 
Date: 23 April 2022 
"""
import numpy as np
import tensorflow as tf
from functools import partial
from typing import Callable, Dict, List, Union

from CORONAnet.math.transforms import reverse_transform_tf


class GeneralizedReweightedLoss():
    """
    Implements Generalized Reweighting Loss function from DisAlign paper
    """
    def __init__(
        self,
        base_loss_fn: Callable,
        bin_edges: List[float],
        event_distribution: List[float],
        scale_parameter: float=0.5,
        label_transforms: Union[List[str], Dict[str, str], str]='log-transform',
    ):
        """
        :param base_loss_fn: Base loss function
        :param bin_edges: Edges of bins to use to divide events into classes
         (based on peak intensity value)
        :param event_distribution: Distribution of events in each bin
        :param scale_parameter: scaling hyperparameter
        """
        self.base_loss_fn = base_loss_fn
        self.bin_edges = bin_edges
        if not isinstance(event_distribution, np.ndarray):
            event_distribution = np.array(event_distribution)
        self.event_distribution = event_distribution
        self.scale_parameter = scale_parameter
        self.label_transforms = label_transforms
        self.calculate_weights()

    def calculate_weights(self):
        """
        Calculate weights for each event category
        """
        summed_ratios = ((1 / self.event_distribution)**self.scale_parameter).sum()
        self.weights = (1 / self.event_distribution)**self.scale_parameter / summed_ratios

    def __call__(
        self, 
        y_true: tf.Tensor, 
        y_pred: tf.Tensor
    ):
        """
        Calculate loss function
        """
        # Reverse the transforms applied to events so we can assign them to their
        # appropriate classes
        y_true_original_vals = reverse_transform_tf(y_true, target_transforms=self.label_transforms)

        # if the rank of the prediction tensor is '1', add a dimension
        y_pred = tf.where(tf.rank(y_pred) == 1, y_pred[..., tf.newaxis], y_pred)
        
        # calculate weighted loss
        weighted_loss = 0.
        for i, (left_edge, right_edge) in enumerate(zip(self.bin_edges[:-1], 
                                                         self.bin_edges[1:])):
            bin_mask = tf.logical_and(
                tf.less(y_true_original_vals, right_edge),
                tf.greater_equal(y_true_original_vals, left_edge),
            )

            # Get events in this category
            y_true_bin = tf.boolean_mask(y_true, bin_mask)
            y_pred_bin = tf.boolean_mask(y_pred, bin_mask)

            # handle case where there are no events in this category in this batch
            bin_loss = tf.cond(tf.shape(y_true_bin)[0] > 0, 
                true_fn=lambda: self.base_loss_fn(y_true_bin, y_pred_bin), 
                false_fn=lambda: tf.constant(0.0, dtype=tf.float32),
            )

            weighted_loss += self.weights[i] * bin_loss

        return tf.reduce_sum(weighted_loss)


def huber_loss(
    y_true: tf.Tensor, 
    y_pred: tf.Tensor, 
    delta: float=1.0
):
    """
    Implementation of huber loss function
    """
    diff = tf.abs(y_true - y_pred)
    loss = tf.where(diff <= delta, (1/2) * diff**2, delta * (diff - (1/2) * delta))
    return loss


def fetch_loss_function(loss_name, **kwargs):
    """
    Grab loss function specified (with additional parameters 
    in kwargs)
    """
    # if no loss function name is provided, return None 
    if loss_name is None:
        return None
    elif loss_name == 'mean-squared-error':
        return tf.keras.losses.MeanSquaredError()
    elif loss_name == 'mean-absolute-error':
        return tf.keras.losses.MeanAbsoluteError()
    elif loss_name == 'huber-loss':
        return partial(huber_loss, delta=kwargs['delta'] if 'delta' in kwargs else 1.0)
    elif loss_name == 'generalized-reweighted-loss':
        return GeneralizedReweightedLoss(
            fetch_loss_function(kwargs['base_loss_function']) if 'base_loss_function' in kwargs else tf.keras.losses.MeanSquaredError(),
            kwargs['bin_edges'] if 'bin_edges' in kwargs else [-np.inf, (10.0 / np.e**2) + 1e-3, 10.0, np.inf],
            kwargs['event_distribution'] if 'event_distribution' in kwargs else [0.98, 0.01, 0.01],
            kwargs['scale_parameter'] if 'scale_parameter' in kwargs else 0.5,
            kwargs['label_transforms'],
        )
    else:
        raise NotImplementedError(f"Loss function {loss_name} not implemented")
