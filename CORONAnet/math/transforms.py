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
    else:
        raise NotImplementedError(f"Transform type {transform_method} not implemented")
