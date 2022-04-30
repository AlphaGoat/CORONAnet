"""
Implements custom training metrics 

Author: Peter Thomas
Date: September 05, 2021 
"""
import numpy as np
from typing import List
import tensorflow as tf
import scipy.spatial.distance as distance
from tensorflow.keras.metrics import Metric

# To prevent division by zero
OFFSET = 1e-8

def argmedian(x: np.ndarray):
    """
    Return index of median value in array 'x'
    """
    if len(x) % 2 == 0:
        return np.argsort(x)[len(x) // 2]
    else:
        return np.argpartition(x, len(x) // 2)[len(x) // 2]


def cosine_similarity(x: np.ndarray, y: np.ndarray):
    """
    Calculate cosine similarity between vectors x and y 
    """
    return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2) + OFFSET)


def cosine_similarity2D(x: np.ndarray, y: np.ndarray):
    """
    Calculates cosine similarity between two 2D matrices x and y 
    """
    # Threshold zero values in both matrices
    x = np.where(x != 0.0, x, OFFSET)
    y = np.where(y != 0.0, y, OFFSET)
    return 1 - distance.cdist(x, y, 'cosine')


def calc_mse_tf(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    Calculate Mean-Squared-Error of true and predicted targets 
    in TensorFlow graph
    """
    return tf.reduce_mean(tf.pow(y_true - y_pred, 2))


def calc_mae_tf(y_true: tf.Tensor, y_pred: tf.Tensor):
    """
    Calculates mean-absolute-error in TensorFlow graph 
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate mean-squared-error 
    """
    return (np.square(y_true - y_pred)).mean()


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate root-mean-squared-error 
    """
    return np.sqrt(np.square(y_true - y_pred).mean())


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculates mean-absolute-error 
    """
    return np.abs(y_true - y_pred).mean()


def stddev_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculates the standard deviation of the absolute error between two arrays 
    """
    return np.std(np.abs(y_true - y_pred))


def pearson_coefficient(x: np.ndarray, y: np.ndarray):
    """
    Determine pearson correlation coefficient between
    two variables x and y 
    """
    x_avg = np.mean(x)
    y_avg = np.mean(y)

    r = ( np.sum((x - x_avg) * (y - y_avg)) /  
        (np.sqrt(np.sum(np.square(x - x_avg))) * np.sqrt(np.sum(np.square(y - y_avg)))) )

    return r


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, thresholds: List=[]):
    """
    Implements confusion matrix with given thresholds
    (intensity at which prediction is regarded as `positive`)
    """
    metrics_dict = {}
    for t in thresholds:

        # mask for positive events
        true_mask = y_true >= t 
        pred_mask = y_pred >= t

        tp = ((pred_mask == True) & (true_mask == True)).sum()
        fp = ((pred_mask == True) & (true_mask == False)).sum()
        tn = ((pred_mask == False) & (true_mask == False)).sum()
        fn = ((pred_mask == False) & (true_mask == True)).sum()

        # precision and recall 
        recall = tpr = tp / (tp + fn)
        precision = tp / (tp + fp)

        fpr = fp / (fp + tn)
        pfa = 1 - precision 

        tss = tpr - fpr 
        f1 = 2 * precision * recall / (precision + recall)


def calc_recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float=10.0): 
    """
    Calculate recall between two target arrays, using the threshold value 
    to distinguish between the 'True' and 'False' class 
    """
    true_mask = y_true >= threshold 
    pred_mask = y_pred >= threshold 

    tp = ((pred_mask == True) & (true_mask == True)).sum()
    fn = ((pred_mask == False) & (true_mask == True)).sum()

    recall = tp / (tp + fn + OFFSET)

    return recall


def calc_precision(y_true: np.ndarray, y_pred: np.ndarray, threshold: float=10.0):
    """
    Calculate precision between two target arrays, using the threshold value 
    to distinguish between 'True' and 'False' classes 
    """
    true_mask = y_true >= threshold 
    pred_mask = y_pred >= threshold 

    tp = ((pred_mask == True) & (true_mask == True)).sum()
    fp = ((pred_mask == True) & (true_mask == False)).sum()

    precision = tp / (tp + fp + OFFSET)

    return precision


def calc_fpr(y_true: np.ndarray, y_pred: np.ndarray, threshold: float=10.0):
    true_mask = y_true >= threshold 
    pred_mask = y_pred >= threshold 

    # false positives
    fp = ((pred_mask == True) & (true_mask == False)).sum()
    tn = ((pred_mask == False) & (true_mask == False)).sum()

    fpr = fp / (fp + tn + OFFSET)

    return fpr
        

def calc_pfa(y_true: np.ndarray, y_pred: np.ndarray, threshold: float=10.0):
    precision = calc_precision(y_true, y_pred, threshold=threshold)
    pfa = 1 - precision

    return pfa


def calc_tss(y_true: np.ndarray, y_pred: np.ndarray, threshold: float=10.0):
    """
    Calculate TSS scores between two target arrays, using the threshold to delineate the 
    'True' and 'False' class 
    """
    tpr = calc_recall(y_true, y_pred, threshold=threshold)

    fpr = calc_fpr(y_true, y_pred, threshold=threshold)

    tss = tpr - fpr

    return tss


def calc_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float=10.0):
    """
    Calculate F1-score between two target arrays, using the threshold to delinear between 
    the 'True' and 'False' classes 
    """
    precision = calc_precision(y_true, y_pred, threshold=threshold)

    recall = calc_recall(y_true, y_pred, threshold=threshold)

    f1 = 2 * precision * recall / (precision + recall + OFFSET)

    return f1


def calc_hss(y_true: np.ndarray, y_pred: np.ndarray, threshold: float=10.0) -> np.ndarray:
    """
    Calculate HSS between two target arrays, using the threshold to delinear between 
    the 'True' and 'False' classes 
    """
    true_mask = y_true >= threshold 
    pred_mask = y_pred >= threshold

    a = ((pred_mask == True) & (true_mask == True)).sum()
    b = ((pred_mask == True) & (true_mask == False)).sum()
    c = ((pred_mask == False) & (true_mask == True)).sum()
    d = ((pred_mask == False) & (true_mask == False)).sum()

    hss = 2 * (a * d - b * c) / ((a + c) * (c + d) + (a + b) * (b + d))

    return hss

class MAE(Metric):
    def __init__(self, name="mae", 
            threshold=None, sep_only_flag=False, **kwargs):
        """
        :param name: (string) Key for metric entry in model history dictionary
        :param threshold: (float) threshold to use to separate SEP events from 
         non-SEP events if sep_only_flag is set 
        :param sep_only_flag: (boolean) flag to set to only track mse for SEP events
        """
        if sep_only_flag:
            name = "sep_mae"

        super(MAE, self).__init__(name=name, **kwargs)

        self.sep_only_flag = sep_only_flag
        if sep_only_flag and threshold is None:
            raise ValueError("ERROR: `threshold` parameter needs to be set if "
            + "sep only flag is set")

        self.threshold = threshold
        self.mae = self.add_weight(name='mae', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.sep_only_flag:
            sep_mask = tf.math.exp(y_true) >= self.threshold
            y_true = tf.boolean_mask(y_true, sep_mask)
            y_pred = tf.boolean_mask(y_pred, sep_mask)

        mae = tf.where(tf.shape(y_true)[0] > 0, calc_mae_tf(y_true, y_pred), 0.0)
        self.mae.assign_add(mae)

    def result(self):
        return self.mae

    def reset_states(self):
        self.mae.assign(0.0)


class MSE(Metric):
    def __init__(self, name="mse", 
            threshold=None, sep_only_flag=False, **kwargs):
        """
        :param name: (string) Key for metric entry in model history dictionary
        :param threshold: (float) threshold to use to separate SEP events from 
         non-SEP events if sep_only_flag is set 
        :param sep_only_flag: (boolean) flag to set to only track mse for SEP events
        """
        if sep_only_flag:
            name = "sep_mse"

        super(MSE, self).__init__(name=name, **kwargs)

        self.sep_only_flag = sep_only_flag
        if sep_only_flag and threshold is None:
            raise ValueError("ERROR: `threshold` parameter needs to be set if "
            + "sep only flag is set")

        self.threshold = threshold
        self.mse = self.add_weight(name='mse', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.sep_only_flag:
            sep_mask = tf.math.exp(y_true) >= self.threshold
            y_true = tf.boolean_mask(y_true, sep_mask)
            y_pred = tf.boolean_mask(y_pred, sep_mask)

        mse = tf.where(tf.shape(y_true)[0] > 0, calc_mse_tf(y_true, y_pred), 0.0)
        self.mse.assign_add(mse)

    def result(self):
        return self.mse

    def reset_states(self):
        self.mse.assign(0.0)


# Classwise Metrics: Calculate metric over entire validation set,
# rather than averaged over batches
class Precision(Metric):
    def __init__(self, name="precision", threshold=0.1, **kwargs):
        # Add threshold to name (so that precision can be evaluated 
        # at multiple threshold values)
        name = name + "_{}".format(threshold)
        super(Precision, self).__init__(name=name, **kwargs)

        self.threshold = threshold
#        self.precision_fn = calc_precision_wrapper(threshold=threshold)

        # Add weights for true and false positives 
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_mask = tf.math.exp(y_true) >= self.threshold 
        pred_mask = tf.math.exp(y_pred) >= self.threshold

        # true positives
        tp = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == True), 
            (true_mask == True)), tf.float32))

        # false positives
        fp = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == True), 
            (true_mask == False)), tf.float32))

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            tp = tf.multiply(tp, sample_weight)
            fp = tf.multiply(fp, sample_weight)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)

    def result(self):
        return (self.true_positives / 
                (self.true_positives + self.false_positives + OFFSET))

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)

class Recall(Metric):
    def __init__(self, name="recall", threshold=0.1, **kwargs):
        # Add threshold to name (so that precision can be evaluated 
        # at multiple threshold values)
        name = name + "_{}".format(threshold)
        super(Recall, self).__init__(name=name, **kwargs)

        self.threshold = threshold
#        self.precision_fn = calc_precision_wrapper(threshold=threshold)

        # Add weights for true and false positives 
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_mask = tf.math.exp(y_true) >= self.threshold 
        pred_mask = tf.math.exp(y_pred) >= self.threshold

        # true positives
        tp = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == True), 
            (true_mask == True)), tf.float32))

        # false negatives
        fn = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == False), 
            (true_mask == True)), tf.float32))

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            tp = tf.multiply(tp, sample_weight)
            fn = tf.multiply(fn, sample_weight)

        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        return (self.true_positives / 
                (self.true_positives + self.false_negatives + OFFSET))

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_negatives.assign(0)

class TSS(Metric):
    def __init__(self, name="tss", threshold=0.1, **kwargs):
        # Add threshold to name (so that precision can be evaluated 
        # at multiple threshold values)
        name = name + "_{}".format(threshold)
        super(TSS, self).__init__(name=name, **kwargs)

        self.threshold = threshold

        # Add weights for true/false positives and true/false negatives 
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_mask = tf.math.exp(y_true) >= self.threshold 
        pred_mask = tf.math.exp(y_pred) >= self.threshold

        # true positives
        tp = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == True), 
            (true_mask == True)), tf.float32))

        # false positives
        fp = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == True), 
            (true_mask == False)), tf.float32))

        # true negatives 
        tn = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == False), 
            (true_mask == False)), tf.float32))

        # false negatives
        fn = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == False), 
            (true_mask == True)), tf.float32))

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            tp = tf.multiply(tp, sample_weight)
            fp = tf.multiply(fp, sample_weight)
            tn = tf.multiply(tn, sample_weight)
            fn = tf.multiply(fn, sample_weight)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.true_negatives.assign_add(tn)
        self.false_negatives.assign_add(fn)

    def result(self):

        tpr =  self.true_positives / (self.true_positives + self.false_negatives + OFFSET)
        fpr = self.false_positives / (self.false_positives + self.true_negatives + OFFSET)

        return tpr - fpr

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_negatives.assign(0)

class F1(Metric):
    def __init__(self, name="f1", threshold=0.1, architecture='torresnet', **kwargs):
        # Add threshold to name (so that precision can be evaluated 
        # at multiple threshold values)
        name = name + "_{}".format(threshold)
        super(F1, self).__init__(name=name, **kwargs)

        self.threshold = threshold
        self.architecture = architecture

        # Add weights for true/false positives and true/false negatives 
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')

        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        if self.architecture == 'BBN':
            alpha = 0.5
            y_pred = alpha * y_pred[0] + (1 - alpha) * y_pred[1]
            y_true = y_true[0]

        true_mask = tf.math.exp(y_true) >= self.threshold 
        pred_mask = tf.math.exp(y_pred) >= self.threshold

        # true positives
        tp = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == True), 
            (true_mask == True)), tf.float32))

        # false positives
        fp = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == True), 
            (true_mask == False)), tf.float32))

        # false negatives
        fn = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == False), 
            (true_mask == True)), tf.float32))

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            tp = tf.multiply(tp, sample_weight)
            fp = tf.multiply(fp, sample_weight)
            fn = tf.multiply(fn, sample_weight)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):

        precision =  self.true_positives / (self.true_positives + self.false_positives + OFFSET)
        recall = self.true_positives / (self.true_positives + self.false_negatives + OFFSET)

        f1 = 2 * precision * recall / (precision + recall + OFFSET)

        return f1

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

class HSS(Metric):
    def __init__(self, name="hss", threshold=0.1, **kwargs):
        # Add threshold to name (so that precision can be evaluated
        # at multiple threshold values)
        name = name + "_{}".format(threshold)
        super(HSS, self).__init__(name=name, **kwargs)

        self.threshold = threshold
        
        # Add weights for true/false positives and true/false negatives 
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_mask = tf.math.exp(y_true) >= self.threshold 
        pred_mask = tf.math.exp(y_pred) >= self.threshold

        # true positives
        tp = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == True), 
            (true_mask == True)), tf.float32))

        # false positives
        fp = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == True), 
            (true_mask == False)), tf.float32))

        # true negatives 
        tn = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == False),
            (true_mask == False)), tf.float32))

        # false negatives
        fn = tf.reduce_sum(tf.cast(tf.logical_and((pred_mask == False), 
            (true_mask == True)), tf.float32))

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            tp = tf.multiply(tp, sample_weight)
            fp = tf.multiply(fp, sample_weight)
            fn = tf.multiply(fn, sample_weight)
            tn = tf.multiply(tn, sample_weight)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
        self.true_negatives.assign_add(tn)

    def result(self):

        a = self.true_positives
        b = self.false_positives 
        c = self.false_negatives 
        d = self.true_negatives 

        hss = 2 * (a * d - b * c) / ((a + c) * (c + d) + (a + b) * (b + d) + OFFSET)

        return hss

    def reset_states(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_negatives.assign(0)
