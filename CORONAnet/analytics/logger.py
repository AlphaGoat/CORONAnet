"""
Logger object to serialize training artifacts in tensorflow summaries format 

Author: Peter Thomas 
Date: 19 April 2022 
"""
import io
import os
import logging
import matplotlib
import pandas as pd
import tensorflow as tf
from datetime import datetime

logger = logging.getLogger(__name__)

class Logger():
    """
    Class for logger and serializing TensorFlow metrics as TF summaries during run 
    """
    def __init__(self,
                 log_dir):
        self.log_dir = log_dir

        # if log directory does not exist, make it 
        os.makedirs(self.log_dir, exist_ok=True)

        # initialize tensorflow summary writer for this log
        self.writer = tf.summary.create_file_writer(log_dir)


    def write_data_to_log(self, data, key=None, step=0):
        """
        Write data to log (general handle for all 
        data object types)
        """
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('r')[0]
            self.write_dict_to_log(data, step=step)
        elif isinstance(data, dict):
            self.write_dict_to_log(data, step=step)
        elif isinstance(data, matplotlib.figure.Figure):
            self.write_plot_to_log(data, key, step=step)
        else:
            try:
                self.write_value_to_log(data, key, step=step)
            except Exception as e:
                logger.info(f"Failed to write {key} of type {type(data)} to log")
                logger.info(f"continuing experiment. {datetime.now()}")


    def write_dict_to_log(self, dictionary, step=0):
        for key in sorted(dictionary):
            self.write_data_to_log(dictionary[key], key=key, step=step)


    def write_plot_to_log(self, fig, name, step=0):
        logger.info(f"Writing plot to tf summary at {self.log_dir}")

        # save matplotlib figure to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)

        # convert PNG buffer to TensorFlow image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0) # batch dim

        with self.writer.as_default():
            tf.summary.image(name, image, step=step)
        self.writer.flush()
        logger.info(f"Plot {name} written to tf summary")


    def write_value_to_log(self, value, key, step=0):
        logger.info(f"Writing {key} to tf summary at {self.log_dir}")
        with self.writer.as_default():
            tf.summary.scalar(key, value, step=step)
        self.writer.flush()
        logger.info("%s: %f", key, value)

    def close(self):
        """
        Close TensorFlow summaries after writing values
        """
        try:
            self.writer.close()
        except RuntimeError as e:
            logger.info(e)
            pass
