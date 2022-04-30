"""
Utilities for parsing tensorflow log files 

Author: Peter Thomas 
Date: 24 April 2022 
"""
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf 
from functools import partial
from tensorflow.python.summary.summary_iterator import summary_iterator

from .plots import temporal_plot, loss_plot
from .series_to_dataframes import temporal_series_from_logs, plots_from_logs


def read_logs(analysis_dir,
              series_functions=[],
              plots_functions=[]):
    """
    Read log files and output a dictionary with all serialized values 
    """
    serialized_log_files = list()

    for root, dirs, files in os.walk(analysis_dir):
        for file in files:
            if 'events.out.tfevents' in file:
                serialized_log_files.append(os.path.join(root, file))

    # iterate through all examples in all files and parse results
    analysis_dict = dict()
    for log_file in serialized_log_files:

        for event in summary_iterator(log_file):
            for v in event.summary.value:
                if v.tag not in analysis_dict:
                    analysis_dict[v.tag] = {
                        "steps": [],
                        "values": [],
                        "wall_times": [],
                    }
                if event.step in analysis_dict[v.tag]['steps']:
                    continue
                analysis_dict[v.tag]['steps'].append(event.step)
                analysis_dict[v.tag]['wall_times'].append(event.wall_time)
                analysis_dict[v.tag]['values'].append(tf.make_ndarray(v.tensor))

    analytics_series = list()
    for func in series_functions:
        df = func(analysis_dict)
        if df is not None:
            analytics_series.append(df)

    analytics_plots = list()
    for func in plots_functions:
        plot = func(analysis_dict)
        if plot is not None:
            analytics_plots.append(plot)

    # concatenate dataframes into one large, mega analytics dataframe
    run_df = pd.DataFrame(columns=['steps'])
    run_df.set_index('steps', inplace=True)

    for series in analytics_series:
        run_df[series.name] = series

    analytics_plots_dict = dict()
    for plot in analytics_plots:
        analytics_plots_dict[plot.name] = plot.get_values()

    return run_df, analytics_plots


def generate_plots(run_df, graphics_save_dir, plot_functions=[]):
    """
    Generate plots in graphics save directory 
    """
    for func_dict in plot_functions:
        func = func_dict["function"]
        plot_name = func_dict["name"]
        fig = func(run_df)
        if fig is not None:
            fig.savefig(os.path.join(graphics_save_dir, plot_name + ".png"), dpi=75)


def save_serialized_plots(run_df, 
                          analytics_plots_dict,
                          graphics_save_dir,
                          plot_keys=[], 
                          best_epoch_metric=None, 
                          higher_score_is_better=True):
    """
    Save plots that were serialized in summaries. If best epoch metric is specified, 
    only save the plot for that epoch of training 

    :param run_df: Dataframe of saved values from summaries file 
    :param plot_keys: Keys for plots that are serialized in dataframe 
    :param best_epoch_metric: Metric to use to determine best epoch of training 
    :param higher_score_is_better: Flag to set if a higher score for the metric is desired 
    """
    if best_epoch_metric is not None and best_epoch_metric in run_df.columns: 
        metric_series = run_df[best_epoch_metric] 
        if higher_score_is_better:
            best_epoch = np.argmax(metric_series)
        else:
            best_epoch = np.argmin(metric_series)

    else:
        # if no evalutaion metric is provided, just plot predictions for 
        # last epoch
        best_epoch = len(run_df)

    for key in plot_keys:
        if key not in analytics_plots_dict:
            continue
        import pdb; pdb.set_trace()
        plot_array = analytics_plots_dict[key].iloc[best_epoch]
        pil_image = Image.fromarray(plot_array)
        pil_image.save(os.path.join(graphics_save_dir, key + ".png"))


def run_analysis(analysis_dir):

    series_functions = [
        lambda x: temporal_series_from_logs(x, "f1_score"),
        lambda x: temporal_series_from_logs(x, "precision"),
        lambda x: temporal_series_from_logs(x, "recall"),
        lambda x: temporal_series_from_logs(x, "hss"),
        lambda x: temporal_series_from_logs(x, "tss"),
        lambda x: temporal_series_from_logs(x, "true_positives"),
        lambda x: temporal_series_from_logs(x, "false_positives"),
        lambda x: temporal_series_from_logs(x, "true_negatives"),
        lambda x: temporal_series_from_logs(x," false_negatives"),
        lambda x: temporal_series_from_logs(x, "mean-absolute-error"),
        lambda x: temporal_series_from_logs(x, "loss/total_loss"),
        lambda x: temporal_series_from_logs(x, "valid_loss/total_loss"),
    ]

    plots_functions = [lambda x: plots_from_logs(x, "prediction_plot")]

    run_df, run_plots_dict = read_logs(analysis_dir, series_functions=series_functions,
                                       plots_functions=plots_functions)

    # create graphics directory inside of analysis directory
    graphics_save_dir = os.path.join(analysis_dir, "graphics")
    os.makedirs(graphics_save_dir, exist_ok=True)

    # Define set of functions to generate plots from data
    plot_functions = [
        {
            "function": loss_plot,
            "name": "loss_plot",
        },
        {
            "function": lambda x: temporal_plot(x, "f1_score", "F1-Score"),
            "name": "f1_plot",
        },
        {
            "function": lambda x: temporal_plot(x, "precision", "Precision"),
            "name": "precision_plot"

        },
        {
            "function": lambda x: temporal_plot(x, "recall", "Recall"),
            "name": "recall_plot"

        },
        {
            "function": lambda x: temporal_plot(x, "hss", "HSS"),
            "name": "hss_plot"

        },
        {
            "function": lambda x: temporal_plot(x, "tss", "TSS"),
            "name": "tss_plot"

        },
        {
            "function": lambda x: temporal_plot(x, "true_positives", "True Positives"),
            "name": "true_positives_plot"

        },
        {
            "function": lambda x: temporal_plot(x, "false_positives", "False Positives"),
            "name": "false_positives_plot"

        },
        {
            "function": lambda x: temporal_plot(x, "false_negatives", "False Negatives"),
            "name": "false_negatives_plot"

        },
        {
            "function": lambda x: temporal_plot(x, "true_negatives", "True Negatives"),
            "name": "true_negatives_plot"

        },
        {
            "function": lambda x: temporal_plot(x, "mean_absolute_error", "Mean Absolute Error"),
            "name": "mean_absolute_error_plot"
        }
    ]

    # generate plots from scalars and save in graphics save directory
    generate_plots(run_df, graphics_save_dir, plot_functions)

    # Now save plots that are serialized in the run dataframe (either generating all plots 
    # or generating plots for best epoch)
    save_serialized_plots(run_df, run_plots_dict, 
                          graphics_save_dir, ["predictions_plot"], "f1_score")


def main_cli(flags):
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--analysis_root_directory', type=str,
                        default=None, help="Root directory to start search for tfrecords log files")

    flags = parser.parse_args()

    run_analysis(flags.analysis_root_directory)
