"""
Transform functions to apply on inputs and outputs 

Author: Peter Thomas 
Date: 2 September 2021
"""
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from scipy.stats import boxcox
import matplotlib.pyplot as plt
from typing import List, Dict, Union


def boxcox_transform(y, lam=1.0, **kwargs):
    return tf.cond(
        tf.not_equal(y, 0.0),
        true_fn=lambda: tf.math.exp(y, lam) / lam,
        false_fn=tf.math.log(y),
    )


def exp_transform(y, scale=1.0, **kwargs):
    """
    Applies exponential function to obtain non-linear_
    targets
    """
    return tf.log(1 + tf.exp((y / scale) - 1))


def apply_transform(y, transform_method=None, **kwargs):
    """
    Apply given transform to y vector 
    """
    if transform_method == 'log-transform':
        y = tf.math.log(y)
    elif transform_method == 'exp-transform':
        y = exp_transform(y, **kwargs)
    elif transform_method == 'boxcox-transform':
        y = boxcox_transform(y, **kwargs)
    elif transform_method == 'longitude-transform':
        return (y + 180.0) / 360.0
    elif transform_method == 'latitude-transform':
        return (y + 90.0) / 180.0
    elif transform_method == 'no-transform' or transform_method is None:
        pass
    else:
        raise NotImplementedError(f"Transform type {transform_method} not implemented")

    return y


def reverse_transform(y, transform_method=None, **kwargs):
    """
    for y that has been projected into a different space by some transform,
    apply reverse trasform and project back into original space 
    """
    if transform_method == 'log-transform':
        return np.exp(y)
    elif transform_method == 'no-transform' or transform_method is None:
        return y 
    elif transform_method == 'longitude-transform':
        return (y * 360.0) - 180.0
    elif transform_method == 'latitude-transform':
        return (y * 180.0) - 90.0
    else:
        raise NotImplementedError(f"Transform type {transform_method} not implemented")


def reverse_transform_tf(
    y: tf.Tensor,
    target_transforms: Union[Dict[str, str], List[str], str]=None,
):
    """
    wrapper for reverse transform that takes
    multiple inputs
    """
    if target_transforms is None:
        return y
    elif isinstance(target_transforms, dict):
        transformed_target_tensors = []
        for i, key in enumerate(target_transforms.keys()):
            transformed_y = apply_reverse_transform_tf(y[..., i], 
                    transform_method=target_transforms[key])
            transformed_target_tensors.append(transformed_y)
        return tf.concat(transformed_target_tensors, axis=-1)
    elif isinstance(target_transforms, list):
        transformed_target_tensors = []
        for i, transform in enumerate(target_transforms):
            transformed_y = apply_reverse_transform_tf(y[..., i], 
                    transform_method=transform)
            transformed_target_tensors.append(transformed_y)
        return tf.concat(transformed_target_tensors, axis=-1)
    elif isinstance(target_transforms, str):
        return apply_reverse_transform_tf(y, transform_method=target_transforms)
    else:
        raise TypeError(f"target_transforms is an unrecognized type {type(target_transforms)}")

def apply_reverse_transform_tf(
    y: tf.Tensor, 
    transform_method: str="log-transform",
    **kwargs
):
    if transform_method == 'log-transform':
        return tf.exp(y)
    elif transform_method == 'no-transform' or transform_method is None:
        return y
    elif transform_method == 'longitude-transform':
        return (y * 360.0) - 180.0
    elif transform_method == 'latitude-transform':
        return (y * 180.0) - 90.0
    else:
        raise NotImplementedError(f"Transform type {transform_method} not implemented")
