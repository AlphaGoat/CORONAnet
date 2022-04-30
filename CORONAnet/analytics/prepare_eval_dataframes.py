"""
Module for creating analytics dataframes for CORONAnet training runs 

Author: Peter Thomas 
Date: 19 April 2022 
"""
import numpy as np
import pandas as pd
from typing import List, Dict

from CORONAnet.math import reverse_transform
from CORONAnet.math.metrics import (
    calc_f1, 
    calc_hss,
    calc_tss, 
    calc_recall, 
    calc_precision, 
    mean_absolute_error,
    pearson_coefficient,
    stddev_absolute_error,
    root_mean_squared_error,
)


def convert_labels_to_df(
    y: np.ndarray, 
    target_labels: List[str]=["peak_intensity"],
):
    """
    Converts label array into a dataframe 

    Args:
        :y: vector of target values to use as dataframe contents 
        :target_labels: List of labels for target values to use as columns of the dataframe

    Returns:
        A pandas dataframe of target values
    """
    return pd.DataFrame(y, columns=target_labels)


def compute_classification_df(
    y_true_df: pd.DataFrame, 
    y_pred_df: pd.DataFrame, 
    target_transform: str=None, 
    sep_threshold: float=10.0, 
    **kwargs
):
    """
    Compute classification statistics for given predictions and
    target values 

    Args:
        :y_true_df: Dataframe of true target values 
        :y_pred_df: Dataframe of predicted target values 
        :target_transform: Transform that was applied to peak intensity label (we 
         will need to reverse the transform to compute the classification metrics)
        :sep_threshold: intensity threshold to use to distinguish SEP events, in pfu

    Returns:
        :classification_table_df: Dataframe containing classification metrics
    """
    # Extract intensity vectors from true and predicted labels
    true_intensity_vector = y_true_df['peak_intensity'].to_numpy(np.float32)
    pred_intensity_vector = y_pred_df['peak_intensity'].to_numpy(np.float32)

    # if transform was applied, reverse transform for all prediction and target values
    true_intensity_vector = reverse_transform(true_intensity_vector, 
                                              transform_method=target_transform, 
                                              **kwargs)
    pred_intensity_vector = reverse_transform(pred_intensity_vector, 
                                              transform_method=target_transform, 
                                              **kwargs)

    # generate classification statistics
    true_mask = true_intensity_vector >= sep_threshold 
    pred_mask = pred_intensity_vector >= sep_threshold

    true_positives = ((true_mask == True) & (pred_mask == True)).sum()
    false_positives = ((true_mask == False) & (pred_mask == True)).sum()
    true_negatives = ((true_mask == False) & (pred_mask == False)).sum()
    false_negatives = ((true_mask == True) & (pred_mask == False)).sum()

    precision = calc_precision(true_intensity_vector, pred_intensity_vector, threshold=sep_threshold)
    recall = calc_recall(true_intensity_vector, pred_intensity_vector, threshold=sep_threshold)
    f1 = calc_f1(true_intensity_vector, pred_intensity_vector, threshold=sep_threshold)
    tss = calc_tss(true_intensity_vector, pred_intensity_vector, threshold=sep_threshold)
    hss = calc_hss(true_intensity_vector, pred_intensity_vector, threshold=sep_threshold)

    class_table_data = dict()
    class_table_data["true_positives"] = [true_positives]
    class_table_data["false_positives"] = [false_positives]
    class_table_data["true_negatives"] = [true_negatives]
    class_table_data["false_negatives"] = [false_negatives]
    class_table_data["precision"] = [precision],
    class_table_data["recall"] = [recall],
    class_table_data["tss"] = [tss],
    class_table_data["f1"] = [f1],
    class_table_data["hss"] = [hss],

    class_table_df = pd.DataFrame.from_dict(class_table_data)

    return class_table_df


def compute_regression_df(
    y_true_df: pd.DataFrame, 
    y_pred_df: pd.DataFrame, 
    target_labels: ["peak_intensity"],
    target_transform: str or List[str] or Dict[str, str]=None, 
    sep_threshold: float=10.0, 
    elevated_intensity_threshold: float=10.0/np.e**2, 
    **kwargs
):
    """
    Compute regression statistics for given predictions and target values 

    Args:
        :y_true_df: Dataframe of true target values 
        :y_pred_df: Dataframe of predicted target values 
        :target_transform: Transform that was applied to peak intensity label (we 
         will need to reverse the transform to compute the classification metrics)
        :sep_threshold: Intensity threshold to use to distinguish SEP events, in pfu
        :elevated_intensity_threshold: Intensity threshold to use to distinguish
         elevated from non-elevated proton events, in pfu

    Returns:
        :regression_table_df: Dataframe with calculated regression metrics
    """
    # if transform was applied, reverse transform for all prediction and target values
    y_true_df = reverse_transform(y_true_df, transform_method=target_transform, **kwargs)
    y_pred_df = reverse_transform(y_pred_df, transform_method=target_transform, **kwargs)

    # Seperate different classes of events (SEP, elevated proton, non-elevated)
    sep_mask = y_true_df['peak_intensity'] >= sep_threshold
    elevated_mask = ((sep_threshold > y_true_df['peak_intensity']) & 
                     (y_true_df['peak_intensity'] > elevated_intensity_threshold))
    
    sep_true = y_true_df[sep_mask]
    sep_pred = y_pred_df[sep_mask]

    elevated_true = y_true_df[elevated_mask]
    elevated_pred = y_pred_df[elevated_mask]
    
    non_elevated_true = y_true_df[~(sep_mask) & ~(elevated_mask)]
    non_elevated_pred = y_pred_df[~(sep_mask) & ~(elevated_mask)]

    # compute metrics for each regression target
    regression_table_data = dict()
    for label in target_labels:
        sep_mae = mean_absolute_error(sep_true[label], sep_pred[label])
        sep_stddev = stddev_absolute_error(sep_true[label], sep_pred[label])

        elevated_mae = mean_absolute_error(elevated_true[label], elevated_pred[label])
        elevated_stddev = stddev_absolute_error(elevated_true[label], elevated_pred[label])

        non_elevated_mae = mean_absolute_error(non_elevated_true[label], non_elevated_pred[label])
        non_elevated_stddev = stddev_absolute_error(non_elevated_true[label],
                                                    non_elevated_pred[label])

        combined_mae = mean_absolute_error(y_true_df[label], y_pred_df[label])
        combined_stddev = stddev_absolute_error(y_true_df[label], y_pred_df[label])

        # calculate pearson correlation between actual and predicted SEP intensities
        r_sep = pearson_coefficient(sep_true[label], sep_pred[label])

        all_elevated_true = np.concatenate([sep_true[label], elevated_true[label]])
        all_elevated_pred = np.concatenate([sep_pred[label], elevated_pred[label]])

        r_elevated = pearson_coefficient(all_elevated_true, all_elevated_pred)

        # Generate table with regression statistics
        regression_table_data[f"{label} (SEP) mean absolute error"] = [sep_mae],
        regression_table_data[f"{label} (SEP) stddev absolute error"] = [sep_stddev],
        regression_table_data[f"{label} (Elevated) mean absolute error"] = [elevated_mae],
        regression_table_data[f"{label} (Elevated) stddev absolute error"] = [elevated_stddev],
        regression_table_data[f"{label} (non-Elevated) mean absolute error"] = [non_elevated_mae]
        regression_table_data[f"{label} (non-Elevated) stddev absolute error"] = [non_elevated_stddev]
        regression_table_data[f"{label} (Combined) mean absolute error"] = [combined_mae]
        regression_table_data[f"{label} (Combined) stddev absolute error"] = [combined_stddev]
        regression_table_data[f"{label} Pearson Coefficient (SEP)"] = [r_sep]
        regression_table_data[f"{label} Pearson Coefficient (Non-Constant)"] = [r_elevated]

    regression_table_df = pd.DataFrame.from_dict(regression_table_data)

    return regression_table_df
