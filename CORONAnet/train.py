"""
Main training script for CORONAnet recurrent convolutional neural network architecture

Author: Peter Thomas
Date: 19 April 2022
"""
import os
import json
import pydash
import logging
import argparse
import datetime
import pyrallis
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Callable, List, Dict

from CORONAnet.config import TrainConfig
from CORONAnet.model import (
    fetch_model,
    fetch_loss_function,
    freeze_feature_extractor,
    reset_regression_head_weights,
)
from CORONAnet.analytics.logger import Logger
from CORONAnet.math import get_downsampled_image_dims
from CORONAnet.analytics.prepare_eval_dataframes import (
    convert_labels_to_df,
    compute_regression_df,
    compute_classification_df, 
)
from CORONAnet.analytics.plots import prediction_plot_handler
from CORONAnet.dataset.DatasetGenerator import DatasetGenerator
from CORONAnet.dataset.parse_functions import get_parse_function

logging.basicConfig(level=logging.INFO)

IMAGE_SHAPE = (512, 512, 1)


def train(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    train_data_generator: DatasetGenerator,
    valid_data_generator: DatasetGenerator,
    log_dir: str,
    current_checkpoint_savepath: str,
    best_checkpoint_savepath: str,
    regression_loss_function: Callable,
    use_autoencoder: bool=False,
    autoencoder_loss_function: Callable=None,
    regression_loss_scale: float=1.0,
    autoencoder_loss_scale: float=1.0,
    event_threshold: float=10.0,
    elevated_intensity_threshold: float=10.0 / np.e**2,
    target_transforms: str or List[str] or Dict[str, str]="log-transform",
    target_labels: List[str]=["peak_intensity"],
    #          warmup_steps=5,
    #          initial_learning_rate=1e-3,
    #          final_learning_rate=1e-4,
    num_train_epochs: int=100):
    """
    Main train function for CORONAnet 

    Args:
        :model: Initialized keras model 
        :optimizer: Gradient optimization function to use for training 
        :train_data_generator: DatasetGenerator object for training inputs 
        :valid_data_generator: DatasetGenerator object for validation inputs 
        :log_dir: Path to directory to store tensorflow summaries objects from training 
        :current_checkpoint_savepath: Path to save current epoch checkpoint
        :best_checkpoint_savepath: Path to save best epoch checkpoint (as determined by 
         eval_metric)
        :regression_loss_function: Function to use for regression loss 
        :use_autoencoder: Flag to set to train with autoencoder branch 
        :regression_loss_scale: Weight to use with regression loss
        :autoencoder_loss_scale: Weight to use with autoencoder loss (if autoencoder branch is
         enabled)
        :event_threshold: Intensity threshold to delineate SEP and non-SEP events (in pfu)
        :elevated_intensity_threshold: Intensity threshold to use to delinear non-SEP events 
         with elevated intensity and those that do not (in pfu)
        :target_transforms: Transform to apply to target values. Can be a string if the same 
         transform is applied to all targets, list of transforms, and dictionary whose keys 
         are the labels for the targets and whose values are the descriptor for the transform 
         (e.g. {'peak_intensity' : 'log-transform'}
        :target_labels: Labels for target values we will be predicting in this training run 
        :num_train_epochs: Number of epochs to use to train model

    Returns:
        None
    """
    
    # initialize logger object to keep track of training metrics
    logger = Logger(log_dir)

    # Create checkpoint directories, if they have not been initialized already
    if current_checkpoint_savepath is not None:
        curr_checkpoint_dir = os.path.dirname(current_checkpoint_savepath) 
        os.makedirs(curr_checkpoint_dir, exist_ok=True)

    if best_checkpoint_savepath is not None:
        best_checkpoint_dir = os.path.dirname(best_checkpoint_savepath)
        os.makedirs(best_checkpoint_dir, exist_ok=True)

    train_dataset = train_data_generator.dataset 
    valid_dataset = valid_data_generator.dataset

    for epoch in range(num_train_epochs):
        print(f"\nepoch {epoch} / {num_train_epochs}")

        count = 0
        best_val_loss = -1.0
        epoch_train_total_loss = 0.0 
        epoch_train_regression_loss = 0.0
        epoch_train_autoencoder_loss = 0.0

        # begin training loop
        for step, (images, targets) in enumerate(train_dataset):

            with tf.GradientTape() as tape:
                pred_results = model(images, training=True)

                # Calculate loss
                regression_loss = regression_loss_function(targets, pred_results[0])

                if use_autoencoder:
                    autoencoder_loss = autoencoder_loss_function(images, pred_results[1])
                    total_loss = regression_loss_scale * regression_loss + autoencoder_loss_scale * autoencoder_loss
                else:
                    total_loss = regression_loss

                # calculate gradients and backpropagate gradients to weights
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_train_total_loss += total_loss
            epoch_train_regression_loss += regression_loss
            count += 1

        # write summary data 
        logger.write_data_to_log(optimizer.lr, "learning_rate", step=epoch)
        logger.write_data_to_log(epoch_train_total_loss / count, 
                                 "loss/total_loss", step=epoch)
        logger.write_data_to_log(epoch_train_regression_loss / count, 
                                "loss/regression_loss", step=epoch)
        if use_autoencoder:
            logger.write_data_to_log(epoch_train_autoencoder_loss / count,
                                     "loss/autoencoder_loss", step=epoch)

        if use_autoencoder:
            print("\n\ntrain_regression_loss: {:7.2f}, ".format(epoch_train_regression_loss / count)
                + "train_autoencoder_loss: {:7.2f}, ".format(epoch_train_autoencoder_loss / count)
                + "train_total_loss:{:7.2f}\n\n".format(epoch_train_total_loss / count))
        else:
            print("\n\ntrain_regression_loss: {:7.2f}, train_total_loss:{:7.2f}\n\n".
                format(epoch_train_regression_loss / count, epoch_train_total_loss / count))

        # perform validation loop
        val_count = 0
        epoch_val_regression_loss, epoch_val_autoencoder_loss, epoch_val_total_loss = 0., 0., 0.
        epoch_val_preds, epoch_val_targets = list(), list()
        for step, (image_data, targets) in enumerate(valid_dataset):

            val_preds = model(image_data, training=False)

            # calculate loss
            val_regression_loss = regression_loss_function(targets, val_preds[0])

            if use_autoencoder:
                val_autoencoder_loss = autoencoder_loss_function(images, val_preds[1])
                total_val_loss = regression_loss_scale * val_regression_loss + \
                        autoencoder_loss_scale * val_autoencoder_loss
            else:
                total_val_loss = val_regression_loss

            val_count += 1
            epoch_val_regression_loss += val_regression_loss 
            epoch_val_total_loss += total_val_loss

            epoch_val_preds.append(val_preds[0].numpy())
            epoch_val_targets.append(targets.numpy())

        epoch_val_preds = np.concatenate(epoch_val_preds)
        epoch_val_targets = np.concatenate(epoch_val_targets)

        # convert predictions and targets into dataframes
        epoch_val_preds_df = convert_labels_to_df(epoch_val_preds, target_labels=target_labels)
        epoch_val_targets_df = convert_labels_to_df(epoch_val_targets, target_labels=target_labels)

        # Output validation metrics
        classification_df = compute_classification_df(epoch_val_targets_df, 
                                                      epoch_val_preds_df,
                                                      target_labels=target_labels,
                                                      event_threshold=event_threshold,
                                                      target_transform=target_transforms)
        regression_df = compute_regression_df(epoch_val_targets_df, 
                                              epoch_val_preds_df,
                                              target_labels=target_labels,
                                              sep_threshold=event_threshold,
                                              elevated_intensity_threshold=elevated_intensity_threshold,
                                              target_transform=target_transforms)
        prediction_plots_dict = prediction_plot_handler(epoch_val_targets_df, 
                                                        epoch_val_preds_df,
                                                        target_labels=target_labels,
                                                        sep_threshold=event_threshold,
                                                        target_transform=target_transforms)

        # write validation metrics to log
        logger.write_data_to_log(classification_df, step=epoch)
        logger.write_data_to_log(regression_df, step=epoch)
        logger.write_data_to_log(prediction_plots_dict, step=epoch)
        logger.write_data_to_log(epoch_val_total_loss / val_count, 
                                 "valid_loss/total_loss", epoch)
        logger.write_data_to_log(epoch_val_regression_loss / val_count, 
                                 "valid_loss/regression_loss", epoch)

        # clear plots 
        plt.close("all")

        if use_autoencoder:
            logger.write_data_to_log(epoch_val_autoencoder_loss / val_count, 
                                     "valid_loss/autoencoder_loss", epoch)

        metrics_to_monitor = dict()
        metrics_to_monitor["F1 Score"] = classification_df['f1'].iloc[0]
        metrics_to_monitor["precision"] = classification_df['precision'].iloc[0]
        metrics_to_monitor["recall"] = classification_df['recall'].iloc[0]
        metrics_to_monitor['TSS'] = classification_df['tss'].iloc[0]
        metrics_to_monitor['HSS'] = classification_df['hss'].iloc[0]

        for label in target_labels:
            metrics_to_monitor[f'{label} mae'] = regression_df[f'{label} (SEP) mean absolute error'].iloc[0][0]

        if use_autoencoder:
            print("\n\nval_regression_loss: {:7.2f}, val_autoencoder_loss: {:7.2f}, total_val_loss:{:7.2f}".
                format(epoch_val_regression_loss / val_count, epoch_val_autoencoder_loss / val_count, 
                       epoch_val_total_loss / val_count))

        else:
            print("\n\nval_regression_loss: {:7.2f}, total_val_loss:{:7.2f}".
                format(epoch_val_regression_loss / val_count, epoch_val_total_loss / val_count))

        for metric in metrics_to_monitor:
            print("{0} : {1:.4f}".format(metric, metrics_to_monitor[metric]))

        # save model checkpoint 
        if current_checkpoint_savepath is not None:
            model.save_weights(current_checkpoint_savepath)

        # if this is the best checkpoint thus far, save as well
        if best_checkpoint_savepath is not None:
            if epoch_val_total_loss < best_val_loss or best_val_loss == -1.:
                best_val_loss = epoch_val_total_loss
                model.save_weights(best_checkpoint_savepath)


@pyrallis.wrap()
def main_cli(flags: TrainConfig):

    # Set visible GPU devices and memory growth
    os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpu_list

    physical_devices = tf.config.experimental.list_physical_devices('GPU')

    # my potato computer needs this to run the model (pwease no insult the potato)
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except RuntimeError as e:
            print(e)

    # if a log directory is provided, save flags to directory
    if flags.log_dir is not None:
        os.makedirs(flags.log_dir, exist_ok=True)
        with open(os.path.join(flags.log_dir, "cli_args.json"), 'w') as f:
            json.dump(flags.__dict__, f, indent=2)

    # Get downsampled image shape
    image_shape = get_downsampled_image_dims(IMAGE_SHAPE, flags.downsample_factor)

    # initialize training data generator
    train_data_generator = DatasetGenerator(
        flags.train_tfrecords_path,
        parse_function=get_parse_function(flags.parse_function, flags.return_filename),
        image_shape=image_shape,
        batch_size=flags.batch_size,
        buffer_size=flags.buffer_size,
        target_labels=["peak_intensity"],
        target_transforms="log-transform",
    )

    # initialize oversampled training data generator (for second stage of training)
    oversampled_train_data_generator = DatasetGenerator(
        flags.train_tfrecords_path,
        parse_function=get_parse_function(flags.parse_function, flags.return_filename),
        image_shape=image_shape,
        batch_size=flags.batch_size,
        buffer_size=flags.buffer_size,
        oversampling_technique=flags.oversampling_technique,
        oversampled_distribution=[
            flags.sep_oversampling_rate,
            1. - flags.sep_oversampling_rate
        ] if flags.sep_oversampling_rate else None,
        target_labels=["peak_intensity"],
        target_transforms="log-transform",
    )

    # initialize validation data generator
    valid_data_generator = DatasetGenerator(
        flags.valid_tfrecords_path,
        parse_function=get_parse_function(flags.parse_function, flags.return_filename),
        image_shape=image_shape,
        batch_size=flags.batch_size,
        buffer_size=flags.buffer_size,
        target_labels=["peak_intensity"],
        target_transforms="log-transform",
    )

    # initialize model
    model = fetch_model(image_shape, flags.model_architecture)

    # if a checkpoint was provided, load model weights to resume training
    if flags.checkpoint_load_path is not None:
        model.load_weights(flags.checkpoint_load_path)

    # get model optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=flags.learning_rate)

    # get regression loss function 
    regression_loss = fetch_loss_function(flags.regression_loss_function)

    # get autoencoder loss function
    autoencoder_loss = fetch_loss_function(flags.autoencoder_loss_function)

    # start model training 
    train(
        model, 
        optimizer, 
        train_data_generator,
        valid_data_generator,
        flags.log_dir,
        flags.current_checkpoint_savepath,
        flags.best_checkpoint_savepath,
        regression_loss,
        autoencoder_loss_function=autoencoder_loss,
        use_autoencoder=True if flags.model_architecture == "VGG16+AE" else None,
        regression_loss_scale=flags.regression_loss_scale,
        autoencoder_loss_scale=flags.autoencoder_loss_scale,
        num_train_epochs=flags.num_representation_training_epochs,
    )

    # Prepare second stage of training by freezing weights of feature extractor 
    # and reinitializing weights in regression head
    freeze_feature_extractor(model)
    reset_regression_head_weights(model)

    # begin second stage of model training
    train(
        model,
        optimizer,
        oversampled_train_data_generator,
        valid_data_generator,
        flags.log_dir,
        flags.current_checkpoint_savepath,
        flags.best_checkpoint_savepath,
        regression_loss,
        num_train_epochs=flags.num_regression_training_epochs,
    )


if __name__ == '__main__':

    main_cli()
