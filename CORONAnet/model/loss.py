"""
Module for implementing loss functions for CORONAnet 

Author: Peter Thomas 
Date: 23 April 2022 
"""
import tensorflow as tf


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
    else:
        raise NotImplementedError(f"Loss function {loss_name} not implemented")
