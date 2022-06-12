"""
Utilities for model weights 

Author: Peter Thomas 
Date: 20 April 2022 
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


def get_gradients(model, optimizer):
    """
    Get gradients of model and return nested dictionary 
    of layers and weights

    :param model: Keras model object 
    :return grad_dict: dictionary of layers and weight gradients 
    """
    grad_dict = {}
    for layer in model.layers:
        for weight in layer.weights:
            grad_dict[layer.name] = {
                weight.name: optimizer.get_gradients(model.total_loss, weight)
            }

    return grad_dict
        

def reset_weights(model):
    """
    Reinitialize weights in model
    """
    for ix, layer in enumerate(model.layers):
        if hasattr(model.layers[ix], 'kernel_initializer') and \
                hasattr(model.layers[ix], 'bias_initializer'):
            weight_initializer = model.layers[ix].kernel_initializer
            bias_initializer = model.layers[ix].bias_initializer 

            old_weights, old_biases = model.layers[ix].get_weights() 

            model.layers[ix].set_weights([
                weight_initializer(shape=old_weights.shape),
                bias_initializer(shape=len(old_biases))])


def reset_trainable_weights(model):
    """
    Reinitialize trainable weights of model 
    """
    for ix, layer in enumerate(model.layers):
        if hasattr(model.layers[ix], 'trainable') and model.layers[ix].trainable:
            if hasattr(model.layers[ix], 'kernel_initializer') and \
                    hasattr(model.layers[ix], 'bias_initializer'):
                weight_initializer = model.layers[ix].kernel_initializer 
                bias_initializer = model.layers[ix].bias_initializer 

                old_weights, old_biases = model.layers[ix].get_weights()

                model.layers[ix].set_weights([
                    weight_initializer(shape=old_weights.shape),
                    bias_initializer(shape=len(old_biases))
                ])


def freeze_weights(model):
    """
    Freeze weights in model 
    """
    for ix, layer in enumerate(model.layers):
        if hasattr(model.layers[ix], 'trainable'):
            model.layers[ix].trainable = False


def reset_layer_weights(layer):
    """
    resets weights in a single layer of model 
    """
    if hasattr(layer, 'kernel_initializer') and \
            hasattr(layer, 'bias_initializer'):
        weight_initializer = layer.kernel_initializer 
        bias_initializer = layer.bias_initializer

        old_weights, old_biases = layer.get_weights()

        layer.set_weights([
            weight_initializer(shape=old_weights.shape),
            bias_initializer(shape=len(old_biases))
        ])


def reset_lstm_layer_weights(lstm_layer):
    """
    Reset weights for LSTM layer
    """
    if hasattr(lstm_layer, 'kernel_initializer') and \
            hasattr(lstm_layer, 'bias_initializer'):
        weight_initializer = lstm_layer.kernel_initializer
        bias_initializer = lstm_layer.bias_initializer

    old_weights1, old_weights2, old_biases = lstm_layer.get_weights()

    lstm_layer.set_weights([
        weight_initializer(shape=old_weights1.shape),
        weight_initializer(shape=old_weights2.shape),
        bias_initializer(shape=len(old_biases)),
    ])

def freeze_layer_weights(layer):
    if hasattr(layer, 'trainable'):
        layer.trainable = False
