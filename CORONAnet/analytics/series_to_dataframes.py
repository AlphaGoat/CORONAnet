"""
Functions that convert series data parsed from tensorflow log files 
and create dataframes from them

Author: Peter Thomas 
Date: 24 April 2022
"""
import numpy as np
import pandas as pd
import tensorflow as tf


def temporal_series_from_logs(analysis_dict, key):
    """
    Extract time series data from analysis dictionary 
    """
    if key not in analysis_dict:
        return None

    data_dict = analysis_dict[key]

    # Get steps and values from dictionary
    steps = np.array(data_dict['steps'])
    vals = np.array(data_dict['values'])

    # order steps and make it the index for the dataframe
    sorted_idxs = np.argsort(steps)

    steps = steps[sorted_idxs]
    vals = vals[sorted_idxs]

    # Make series object 
    temporal_series = pd.Series(vals, index=steps, name=key) 

    return temporal_series


def plots_from_logs(analysis_dict, key):
    """
    Extract plots from logs and return a series that contains 
    all plot information with time steps as the index 
    """
    if key not in analysis_dict:
        return None

    data_dict = analysis_dict[key] 

    # get steps from dictionary
    steps = np.array(data_dict['steps'])

    # iterate over plots and decode images 
    plots = list()
    for buf in data_dict['values']:
        image = tf.image.decode_image(buf[2])
        plots.append(image.numpy())

    # Make a series of the plot
    plot_series = pd.Series(plots, index=steps, name=key)

    return plot_series
