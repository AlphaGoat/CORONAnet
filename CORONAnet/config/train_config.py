"""
Train configuration file. Define all the different configurations for training here

Author: Peter Thomas 
Date: 23 April 2022 
"""
import pyrallis
import numpy as np
from typing import List
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """
    Training configuration for CORONAnet 
    """
    # Path to train tfrecord file or directory with training tfrecords
    train_tfrecords_path: str = None 

    # Path to validation tfrecord file or directory with validation tfrecords
    valid_tfrecords_path: str = None 

    # Model architecture to use for training
    model_architecture: str = "VGG16+AE"

    # Rate at which to oversample SEP events
    sep_oversampling_rate: float = None

    # Rate at which to oversample high speed and large width non-SEP events
    hi_speed_large_width_non_sep_oversampling_rate: float = None

    # Technique to use to perform oversampling with
    oversampling_technique: str = None

    # Parse function to use for serialized tfrecord example protos 
    parse_function: str = "multi_frame_parse"

    # Boolean flag to set to return tensor of image filenames when parsing data 
    return_filename: bool = False

    # Number of examples to fill prefetch buffer with 
    buffer_size: int = 1

    # Data batch size to use for training 
    batch_size: int = 1

    # Factor to downsample images by (to nearest power of 2)
    downsample_factor: int = 1

    # Number of epochs to train model during first stage. 
    num_representation_training_epochs: int = 100

    # Number of epochs to train model during second stage.
    num_regression_training_epochs: int = 50

    # Learning rate to use for training 
    learning_rate: float = 1e-4

    # Loss function to use for regression branch 
    regression_loss_function: str = "mean-squared-error"

    # Loss function to use for representation branch
    autoencoder_loss_function: str = "mean-squared-error"

    # Weight for regression loss
    regression_loss_scale: float = 1.0

    # Weight for autoencoder loss (if it is being used)
    autoencoder_loss_scale: float = 1.0

    # Evaluation metric to use to determine best epoch of training 
    eval_metric: str = "valid-loss"

    # Threshold intensity for SEP events (in pfu)
    event_threshold: float = 10.0

    # List of GPU devices to use for model training
    gpu_list: str = '0'

    # Path to save current checkpoint
    current_checkpoint_savepath: str = None

    # Path to save best checkpoint
    best_checkpoint_savepath: str = None

    # Path to weights to load to resume training from earlier checkpoint
    checkpoint_load_path: str = None

    # Path to save log files for training 
    log_dir: str = None


#@dataclass
#def ModelConfig:
#    pass

class DatasetConfig:
    # Labels for targets we would like to predict for this model run
    target_labels: List[str] = ["peak_intensity"]

    # Intensity threshold to distinguish SEP and non-SEP events, in pfu 
    sep_threshold: float = 10.0

    # Intensity threshold to distinguish elevated non-SEP events, in pfu
    elevated_intensity_threshold: float = 10.0 / np.e**2

@dataclass
class LossConfig:
    pass


@dataclass
class ExperimentConfig:
    pass
