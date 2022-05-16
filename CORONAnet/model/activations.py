"""
File for custom activation functions

Author: Peter Thomas
Date: 16 May 2022
"""
import tensorflow as tf

def LeakyReLU(alpha=0.2):
    return lambda x: tf.keras.backend.maximum(alpha * x, x)
