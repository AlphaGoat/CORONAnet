import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
from CORONAnet.math import reverse_transform


def prediction_plot_handler(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    target_labels: List[str]=["peak_intensity"],
    target_transform: str or List[str] or Dict[str, str]='log-transform',
    sep_threshold: float=10.0,
    elevated_intensity_threshold: float=np.e**2
):
    """
    Generate prediction plots for all target labels and 
    return in dictionary

    Args:
        :y_true_df: dataframe containing truth target values 
        :y_pred_df: dataframe containing predicted target values
        :target_labels: list of labels for each target we are trying to predict
        :target_transform: transform to apply to target values. 

    Returns:
        :prediction_plots_dict: Dictionary with keys referring to the target being 
         plotted and whose values are the matplotlib figures for the prediction plots
    """
    def _get_transform(label):
        if isinstance(target_transform, str):
            return target_transform 
        elif isinstance(target_transform, list):
            idx = target_labels.index(label)
            return target_transform[idx]
        elif isinstance(target_transform, dict):
            return target_transform[label]

    prediction_plots_dict = dict()
    for i, label in enumerate(target_labels):

        fig = plt.figure(i)
        if 'peak_intensity' in y_true_df.columns and 'peak_intensity' in y_pred_df.columns:
            fig = generate_categorized_prediction_plot(
                y_true_df, 
                y_pred_df,
                target_label=label,
                target_transform=_get_transform(label),
                sep_threshold=sep_threshold,
                elevated_intensity_threshold=elevated_intensity_threshold,
                fig=fig,
            )
        else:
            true_target_vector = y_true_df[label]
            pred_target_vector = y_pred_df[label]
            fig = generate_prediction_plot(
                true_target_vector,
                pred_target_vector,
                target_label=label,
                target_transform=_get_transform(label),
                fig=fig
            )

        prediction_plots_dict[label + "_prediction_plot"] = fig

    return prediction_plots_dict


def generate_categorized_prediction_plot(
    y_true_df: pd.DataFrame, 
    y_pred_df: pd.DataFrame, 
    target_label: str='peak_intensity',
    target_transform: str='log-transform', 
    sep_threshold: float=10.0,
    elevated_intensity_threshold: float=10.0 / np.e**2, 
    fig: plt.Figure=None, 
    **kwargs
):
    """
    Generate intensity prediction plots with predicted intensity as x-axis and
    true intensity as y-axis (in log-scale)

    Args:
        :y_true: array of true target values
        :y_pred: array of predicted target values
        :target_label: label of target we would like to plot
        :target_transform: transform that was applied to target during training (if the
         transform was anything but a log transform, we will need to reverse the transform
         and apply a log transform for plotting
        :sep_threshold: Threshold to use to distinguish SEP events from non-SEP events, in pfu
        :elevated_intensity_threshold: Threshold to distinguish elevated intensity events from
         non-elevated intensity protonh events, in pfu

    Returns:
        :fig: plot of predictions
    """
    y_true = y_true_df[target_label]
    y_pred = y_pred_df[target_label]

    # reverse transform and apploy log to predictions and targets unless they are already
    # in log scale
    if target_transform != 'log-transform':
        y_true = reverse_transform(y_true, transform_method=target_transform, **kwargs)
        y_pred = reverse_transform(y_pred, transform_method=target_transform, **kwargs)

        y_true = np.log(y_true)
        y_pred = np.log(y_pred)

    log_sep_threshold = np.log(sep_threshold)
    log_elevated_threshold = np.log(elevated_intensity_threshold)

    # separate sep events and elevated intensity events
    sep_mask = y_true_df['peak_intensity'] >= log_sep_threshold
    elevated_mask = (log_sep_threshold > y_true_df['peak_intensity']) & (y_true['peak_intensity'] > log_elevated_threshold)

    sep_true = y_true[sep_mask]
    sep_pred = y_pred[sep_mask]

    elevated_true = y_true[elevated_mask]
    elevated_pred = y_pred[elevated_mask]

    non_elevated_true = y_true[(~sep_mask) & (~elevated_mask)]
    non_elevated_pred = y_pred[(~sep_mask) & (~elevated_mask)]

    # plot predictions (color coding different categories of objects:
    # Red: SEP-events
    # Green: elevated CME events
    # Blue: non-elevated CME events
    if fig is None:
        fig = plt.figure()

    ax = plt.subplot(111)
    ax.scatter(non_elevated_true, non_elevated_pred, color='blue')
    ax.scatter(elevated_true, elevated_pred, color='green', edgecolor='black')
    ax.scatter(sep_true, sep_pred, color='red', edgecolor='black')

    # some additional handling depending on what target we are plotting
    if target_label == 'peak_intensity':
        # plot thresholds
        ax.axhline(y=log_sep_threshold, color='black', linestyle='--')
        ax.axvline(x=log_sep_threshold, color='black', linestyle='--')

        ax.set_xlim([-2.0, 10.0])

    elif target_label == 'threshold_time':
        # plot thresholds
        ax.axhline(y=np.log(4500.0), color='black', linestyle='--')
        ax.axvline(x=np.log(4500.0), color='black', linestyle='--')

        ax.set_xlim([-2.0, 10.0])

    # plot line of equality 
    lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    plt.xlabel(f"Observed Proton Log {''.join(list(map(lambda x: x.title(), target_label.split('_'))))}") 
    plt.ylabel("Predicted Proton Log {''.join(list(map(lambda x: x.title(), target_label.split('_'))))}")

    return fig


def generate_prediction_plot(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    target_label: str='peak_intensity',
    target_transform: str='log-transform', 
    fig: plt.Figure=None, 
    **kwargs
):
    """
    Generate intensity prediction plots with predicted intensity as x-axis and
    true intensity as y-axis (in log-scale)

    Args:
        :y_true: array of true target values
        :y_pred: array of predicted target values
        :target_label: label of target we would like to plot
        :target_transform: transform that was applied to target during training (if the
         transform was anything but a log transform, we will need to reverse the transform
         and apply a log transform for plotting
        :sep_threshold: Threshold to use to distinguish SEP events from non-SEP events, in pfu
        :elevated_intensity_threshold: Threshold to distinguish elevated intensity events from
         non-elevated intensity protonh events, in pfu

    Returns:
        :fig: plot of predictions
    """
    # reverse transform and apploy log to predictions and targets unless they are already
    # in log scale
    if target_transform != 'log-transform':
        y_true = reverse_transform(y_true, transform_method=target_transform, **kwargs)
        y_pred = reverse_transform(y_pred, transform_method=target_transform, **kwargs)

        if (target_transform != 'no-transform' or target_transform is not None
                or target_transform != 'longitude-transform' or target_transform != 'latitude-transform'):
            y_true = np.log(y_true)
            y_pred = np.log(y_pred)


    # plot predictions
    if fig is None:
        fig = plt.figure()

    ax = plt.subplot(111)
    ax.scatter(y_true, y_pred, color='blue')

    # plot line of equality 
    lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    plt.xlabel(f"Observed Log {''.join(list(map(lambda x: x.title(), target_label.split('_'))))}") 
    plt.ylabel("Predicted Log {''.join(list(map(lambda x: x.title(), target_label.split('_'))))}")

    return fig


def loss_plot(run_df: pd.DataFrame):
    """
    Plot training and validation loss for a run 
    """
    if 'loss/total_loss' and 'valid_loss/total_loss' not in run_df.columns:
        return None

    train_loss = run_df['loss/total_loss']
    valid_loss = run_df['valid_loss/total_loss']
    steps = run_df.index

    fig = plt.figure()

    plt.plot(steps, train_loss)
    plt.plot(steps, valid_loss)
    plt.title("Training Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(['train', 'validation'], loc='upper left')

    return fig


def temporal_plot(run_df: pd.DataFrame, key: str, title: str=""):
    """
    Plot data from logs as time-series 
    """
    if key not in run_df.columns:
        return None

    steps = run_df.index
    values = run_df[key]

    fig = plt.figure()

    plt.plot(steps, values)
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel(key)

    return fig
