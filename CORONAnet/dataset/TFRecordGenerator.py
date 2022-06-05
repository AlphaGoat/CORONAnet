"""
Utilties for writing serialized TF Record files for CORONAnet

Author: Peter Thomas
Date: 30 April 2022
"""
import os
import csv
import cv2
import glob
import shutil 
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf 
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta

from SEPnet.utils.match_events import (
        jd_to_date,
        sep_data_to_df,
        correlate_cme_to_proton_event,
)
from SEPnet.math.partition_funcs import (
    normalize_partitions,
    regionalStratifiedBinning, 
    regionalStratifiedBinning2D
)
from SEPnet.dataset.data_generator import get_date_labels_by_minute

from CORONAnet import TARGET_LABELS
from CORONAnet.utils import ask_for_confirmation


def _bytes_feature(value):
    """
    Returns bytes_list from a string / byte
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """
    returns a float_list from a float / double
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """
    Returns an int64 list from a bool / enum / int / uint
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_basename(path):
    """
    Get the base filename from path 
    """
    return os.path.splitext(os.path.basename(path))[0]


def write_sequence_to_disk(sequence, timestamp, savepath):
    pass


def to_batch(l, n):
    """
    Convert a list l to a list of lists, each with size n (until there 
    are less than n frames left)
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]


def write_partition_to_tfrecord(cme_event_df: pd.DataFrame,
                                sequences_array: np.array,
                                partition_name: str,
                                partition_path: str,
                                image_shape: Tuple[int] or List[int]=None,
                                target_labels: List[str]=['peak_intensity'],
                                num_examples_per_tfrecord: int=4):
    """
    Write partition to tfrecord 
    """
    # Get the number of tfrecords we are going to have to write for this partition
    num_tfrecords_to_write = len(cme_event_df) // num_examples_per_tfrecord

    done_writing = False
    current_record = 0
    num_examples_written = 0
    while not done_writing:

        tfrecord_filename = f'{partition_name}_{current_record}.tfrecord'
        print(f"writing {tfrecord_filename} to file...")
        with tf.io.TFRecordWriter(os.path.join(partition_path,
                  tfrecord_filename)) as writer:

            for _ in range(num_examples_per_tfrecord):

                if num_examples_written == len(cme_event_df):
                    done_writing = True 
                    break

                sequence = sequences_array[num_examples_written]
                cme_label = cme_event_df.iloc[num_examples_written]

                # if the sequence is empty, continue to next example
                if len(sequence) == 0:
                    num_examples_written += 1
                    continue

                example = serialize_sequence_data(sequence, cme_label, 
                                                  image_shape=image_shape,
                                                  target_labels=target_labels)

                writer.write(example.SerializeToString())

                num_examples_written += 1
            
        current_record += 1


def prepare_10MeV_proton_event_dataframes(ten_mev_event_file,
                                          ten_mev_event_re2_file):
    """
    Prepare dataframes for SEPs and elevated proton intensity events 
    with 10 MeV datafiles (using 10 and 10re2 thresholds)
    """
    ten_mev_events_df = pd.read_csv(ten_mev_event_file)
    ten_mev_events_re2_df = pd.read_csv(ten_mev_event_re2_file)

    ten_mev_events_df = ten_mev_events_df.reset_index(drop=True)
    ten_mev_events_re2_df= ten_mev_events_re2_df.reset_index(drop=True)

    ten_mev_events_df['datetime'] = pd.to_datetime(
        ten_mev_events_df['datetime'], format="%Y-%m-%d %H:%M:%S"
    )

    ten_mev_events_re2_df['datetime'] = pd.to_datetime(
        ten_mev_events_re2_df['datetime'], format="%Y-%m-%d %H:%M:%S"
    )

    # remove events in the 10 MeV dataframe that are below a peak intensity of 10.0
    sep_peak_indices = ten_mev_events_df[2::4][
        ten_mev_events_df[2::4]['Intensity'] >= 10.0
    ].index
    sep_event_indices = list()
    for idx in sep_peak_indices:
        sep_event_indices.extend([i for i in range(idx-2, idx+2)])

    ten_mev_sep_events_df = ten_mev_events_df.loc[sep_event_indices]

    # If there are SEP events in the ten mev re2 events datafile that 
    # are not in the original SEP event file, link them
    sep_re2_peak_indices = ten_mev_events_re2_df[2::4][
            ten_mev_events_re2_df[2::4]['Intensity'] >= 10.0].index
    indice_copies = sep_re2_peak_indices.copy()
    for idx in sep_re2_peak_indices:
        sep_event = ten_mev_events_re2_df.loc[idx]
        ten_mev_matched = ten_mev_sep_events_df[2::4][
            ten_mev_sep_events_df[2::4]['datetime'] == sep_event['datetime']
        ]
        if len(ten_mev_matched) > 0:
            index_mask = indice_copies != idx
            indice_copies = indice_copies[index_mask]

    unique_sep_indices = []
    for idx in indice_copies:
        unique_sep_indices.extend([i for i in range(idx-2, idx+2)])

    # pull all re SEP events that are not in original SEP dataframe
    re2_sep_events_df = ten_mev_events_re2_df.loc[unique_sep_indices]
    ten_mev_re2_events_df = ten_mev_events_re2_df.drop(index=unique_sep_indices)

    ten_mev_sep_events_df = pd.concat([ten_mev_sep_events_df, re2_sep_events_df])

    # remove all remaining SEP events in re2 dataframe (the ones that are remaining
    # should already be included in the ten_mev_sep dataframe)
    re_sep_peak_indices = ten_mev_re2_events_df[2::4][
        ten_mev_re2_events_df[2::4]['Intensity'] >= 10.0
    ].index
    
    re_sep_full_indices = []
    for idx in re_sep_peak_indices:
        re_sep_full_indices.extend([i for i in range(idx - 2, idx + 2)])
    
    # drop indices of SEP events in re2 dataframe
    ten_mev_re2_events_df = ten_mev_re2_events_df.drop(index=re_sep_full_indices)

    return ten_mev_sep_events_df, ten_mev_re2_events_df


def partition_dataset(sep_correlated_cme_df,
                      non_sep_correlated_cme_df, 
                      uncorrelated_cme_df,
                      train_partition=0.5, 
                      valid_partition=0.2, 
                      test_partition=0.3,
                      partition_method='random'):
    """
    Partition CME events into training, validation, and test sets
    """
    if partition_method == 'random':
        # randomly sort events into train, validation, and test sest 
        # (different types of events are treated differently. for example,
        # 70% of SEP events will be placed in train and 30% to test, etc.
        train_sep_idx = int(0.60 * len(sep_correlated_cme_df))
        valid_sep_idx = int((0.60 + 0.2) * len(sep_correlated_cme_df))

        train_sep_df, valid_sep_df, test_sep_df = np.split(
            sep_correlated_cme_df.sample(frac=1),
            [train_sep_idx, valid_sep_idx]
        )

        train_non_sep_idx = int(train_partition * len(non_sep_correlated_cme_df))
        valid_non_sep_idx = train_non_sep_idx + int(valid_partition * 
                len(non_sep_correlated_cme_df))

        train_non_sep_df, valid_non_sep_df, test_non_sep_df = np.split(
            non_sep_correlated_cme_df.sample(frac=1),
            [train_non_sep_idx, valid_non_sep_idx]
        )

        train_uncorr_idx = int(train_partition * len(uncorrelated_cme_df))
        valid_uncorr_idx = train_uncorr_idx + int(valid_partition * 
                len(uncorrelated_cme_df))

        train_uncorr_df, valid_uncorr_df, test_uncorr_df = np.split(
            uncorrelated_cme_df.sample(frac=1),
            [train_uncorr_idx, valid_uncorr_idx]
        )

        train_df = pd.concat([train_sep_df, train_non_sep_df, train_uncorr_df])
        valid_df = pd.concat([valid_sep_df, valid_non_sep_df, valid_uncorr_df])
        test_df = pd.concat([test_sep_df, test_non_sep_df, test_uncorr_df])

        train_df = train_df.sample(frac=1)
        valid_df = valid_df.sample(frac=1)
        test_df = test_df.sample(frac=1)

    elif partition_method == 'sep-balanced':
        # balance by peak intensity
        _, bin_edges = pd.qcut(sep_correlated_cme_df['peak_intensity'], q=10, retbins=True)
        bin_edges = [(left, right) for left, right in zip(bin_edges[:-1], bin_edges[1:])]

        all_events = []
        for i, (left_edge, right_edge) in enumerate(bin_edges):
            if i == 0:
                events_df = sep_correlated_cme_df[
                    (left_edge <= sep_correlated_cme_df['peak_intensity']) &
                    (sep_correlated_cme_df['peak_intensity'] <= right_edge)
                ]

            else:
                events_df = sep_correlated_cme_df[
                    (left_edge < sep_correlated_cme_df['peak_intensity']) &
                    (sep_correlated_cme_df['peak_intensity'] <= right_edge)
                ]

            all_events.append(events_df)

        # split events into training, validation, and test partition
        all_train_events, all_valid_events, all_test_events = [], [], []
        for bin_events in all_events:

            train_idx = int(0.60 * len(bin_events))
            valid_idx = int((0.60 + 0.2) * len(bin_events))
            train_events, valid_events, test_events = np.split(
                bin_events.sample(frac=1), [train_idx, valid_idx]
            )

            all_train_events.append(train_events)
            all_valid_events.append(valid_events)
            all_test_events.append(test_events)

        train_sep_df = pd.concat(all_train_events)
        valid_sep_df = pd.concat(all_valid_events)
        test_sep_df = pd.concat(all_test_events)

        train_sep_df = train_sep_df.sample(frac=1)
        valid_sep_df = valid_sep_df.sample(frac=1)
        test_sep_df = test_sep_df.sample(frac=1)

        # split rest of CME events evenly
        non_sep_events_df = pd.concat([non_sep_correlated_cme_df, uncorrelated_cme_df])

        train_idx = int(train_partition * len(non_sep_events_df))
        valid_idx = train_idx + int(valid_partition * len(non_sep_events_df))

        train_non_sep_events_df, valid_non_sep_events_df,\
            test_non_sep_events_df = np.split(non_sep_events_df.sample(frac=1), [train_idx, valid_idx])

        # combine sep events and non-sep events
        train_df = pd.concat([train_sep_df, train_non_sep_events_df])
        valid_df = pd.concat([valid_sep_df, valid_non_sep_events_df])
        test_df = pd.concat([test_sep_df, test_non_sep_events_df])

        train_df = train_df.sample(frac=1)
        valid_df = valid_df.sample(frac=1)
        test_df = test_df.sample(frac=1)

    elif partition_method == 'chronological':

        # sort events by time and divide into train, valid, and test partitions
        # (so first events go into train partition and later events go to test 
        # partition)
        events_df = pd.concat([sep_correlated_cme_df, non_sep_correlated_cme_df,
            uncorrelated_cme_df])

        # convert all event timestamps to datetime objects
        events_df['donki_date'] = events_df['donki_date'].map(
            lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S") if  
            isinstance(t, str) else t)

        events_df = events_df.sort_values(by='donki_date')

        train_idx = int(train_partition * len(events_df))
        valid_idx = train_idx + int(valid_partition * len(events_df))

        train_df, valid_df, test_df = np.split(events_df, [train_idx, valid_idx])

        train_df = train_df.sample(frac=1)
        valid_df = valid_df.sample(frac=1)
        test_df = test_df.sample(frac=1)

    elif partition_method == 'sep-and-non-sep-balanced':

        # balance SEPs by peak intensity
        _, bin_edges = pd.qcut(sep_correlated_cme_df['peak_intensity'], q=10, retbins=True)
        bin_edges = [(left, right) for left, right in zip(bin_edges[:-1], bin_edges[1:])]

        all_events = []
        for i, (left_edge, right_edge) in enumerate(bin_edges):
            if i == 0:
                events_df = sep_correlated_cme_df[
                    (left_edge <= sep_correlated_cme_df['peak_intensity']) &
                    (sep_correlated_cme_df['peak_intensity'] <= right_edge)
                    ]

            else:
                events_df = sep_correlated_cme_df[
                    (left_edge < sep_correlated_cme_df['peak_intensity']) &
                    (sep_correlated_cme_df['peak_intensity'] <= right_edge)
                    ]

            all_events.append(events_df)

        # split events into training, validation, and test partition
        all_train_events, all_valid_events, all_test_events = [], [], []
        for bin_events in all_events:
            train_idx = int(0.60 * len(bin_events))
            valid_idx = int((0.60 + 0.2) * len(bin_events))
            train_events, valid_events, test_events = np.split(
                bin_events.sample(frac=1), [train_idx, valid_idx]
            )

            all_train_events.append(train_events)
            all_valid_events.append(valid_events)
            all_test_events.append(test_events)

        train_sep_df = pd.concat(all_train_events)
        valid_sep_df = pd.concat(all_valid_events)
        test_sep_df = pd.concat(all_test_events)

        train_sep_df = train_sep_df.sample(frac=1)
        valid_sep_df = valid_sep_df.sample(frac=1)
        test_sep_df = test_sep_df.sample(frac=1)

        # divide non-SEP events into four cases
        # case a) events >= 1000.0 speed and >= 40.0 half angle
        # case b) events >= 1000.0 speed and < 40.0 half angle
        # case c) events < 1000.0 speed and >= 40.0 half angle
        # case d) events < 1000.0 speed and < 40.0 half angle
        non_sep_events_df = pd.concat([non_sep_correlated_cme_df, uncorrelated_cme_df])
        combined_partition = train_partition + valid_partition
        train_in_combined_split, valid_in_combined_split = normalize_partitions(
            train_partition,
            valid_partition
        )

        case_a_df = non_sep_events_df[
            ((non_sep_events_df['donki_speed'] > 1000.0) & 
             (non_sep_events_df['donki_ha'] > 40.0))
        ].sample(frac=1)

        case_b_df = non_sep_events_df[
            ((non_sep_events_df['donki_speed'] > 1000.0) & 
             (non_sep_events_df['donki_ha'] <= 40.0))
        ].sample(frac=1)

        case_c_df = non_sep_events_df[
            ((non_sep_events_df['donki_speed'] <= 1000.0) & 
             (non_sep_events_df['donki_ha'] > 40.0))
        ].sample(frac=1)

        case_d_df = non_sep_events_df[
            ((non_sep_events_df['donki_speed'] <= 1000.0) & 
             (non_sep_events_df['donki_ha'] <= 40.0))
        ].sample(frac=1)

        # perform 2d stratified regional binning on case-a events using donki speed and half angle
        (combined_a_df, test_a_df), \
            (combined_a_counts, test_a_counts), \
            (case_a_speed_bins, case_a_ha_bins) = regionalStratifiedBinning2D(
            case_a_df, feature_1='donki_speed', feature_2='donki_ha',
            num_feature1_bins=4, num_feature2_bins=4, region1_dims=2,
            region2_dims=2, train_partition=combined_partition,
            test_partition=test_partition, retedges=True
        )
        train_a_idx = int(train_in_combined_split * len(combined_a_df))
        combined_a_df = combined_a_df.sample(frac=1)
        train_a_df, valid_a_df = np.split(combined_a_df, [train_a_idx])
        train_a_df = train_a_df.sample(frac=1)
        valid_a_df = valid_a_df.sample(frac=1)
        test_a_df = test_a_df.sample(frac=1)

        # perform 1d stratified regional binning on case-b events using donki speed
        (combined_b_df, test_b_df), \
            (combined_b_counts, test_b_counts), \
            case_b_speed_bins = regionalStratifiedBinning(
            case_b_df, feature='donki_speed', num_feature_bins=16,
            region_dims=4, train_partition=combined_partition,
            test_partition=test_partition, retedges=True,
        )
        combined_b_df = combined_b_df.sample(frac=1)
        train_b_idx = int(train_in_combined_split * len(combined_b_df))
        train_b_df, valid_b_df = np.split(combined_b_df, [train_b_idx])
        train_b_df = train_b_df.sample(frac=1)
        valid_b_df = valid_b_df.sample(frac=1)
        test_b_df = test_b_df.sample(frac=1)

        # perform 1d stratified regional binning on case-c events using donki half-angle
        (combined_c_df, test_c_df), \
            (combined_c_counts, test_c_counts), \
                case_c_ha_bins = regionalStratifiedBinning(
            case_c_df, feature='donki_ha', num_feature_bins=80,
            region_dims=20, train_partition=combined_partition,
            test_partition=test_partition, retedges=True,
        )
        combined_c_df = combined_c_df.sample(frac=1)
        train_c_idx = int(train_in_combined_split * len(combined_c_df))
        train_c_df, valid_c_df = np.split(combined_c_df, [train_c_idx])
        train_c_df = train_c_df.sample(frac=1)
        valid_c_df = valid_c_df.sample(frac=1)
        test_c_df = test_c_df.sample(frac=1)

        # we don't care so much about case d events, so split randomly
        train_idx = int(train_partition * len(case_d_df))
        valid_idx = train_idx + int(valid_partition * len(case_d_df))

        train_d_df, valid_d_df, test_d_df = np.split(case_d_df, [train_idx, valid_idx])
        train_d_df = train_d_df.sample(frac=1)
        valid_d_df = valid_d_df.sample(frac=1)
        test_d_df = test_d_df.sample(frac=1)

        # combined non-SEP events
        train_non_sep_df = pd.concat([train_a_df, train_b_df, train_c_df, train_d_df]).sample(frac=1)
        valid_non_sep_df = pd.concat([valid_a_df, valid_b_df, valid_c_df, valid_d_df]).sample(frac=1)
        test_non_sep_df = pd.concat([test_a_df, test_b_df, test_c_df, test_d_df]).sample(frac=1)

        # combine sep events and non-sep events
        train_df = pd.concat([train_sep_df, train_non_sep_df])
        valid_df = pd.concat([valid_sep_df, valid_non_sep_df])
        test_df = pd.concat([test_sep_df, test_non_sep_df])

        train_df = train_df.sample(frac=1)
        valid_df = valid_df.sample(frac=1)
        test_df = test_df.sample(frac=1)

    else:
        raise ValueError("Partition method {} unrecognized".format(partition_method))

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    return train_df, valid_df, test_df


def partition_data(corona_df,
                   chronological_sort=False,
                   splits=[]):
    """
    Sort data, either randomly or chronologically, into partitons 
    provided (if none, then just return corona df as is)
    """
    if not splits:
        return [corona_df]

    # sort data by timestamp if partitioning chronologically
    if chronological_sort:
        corona_df = corona_df.sort_values('datetime').reset_index(drop=True)

    else:
        corona_df = corona_df.sample(frac=1)

    # divide into number of partitions
    num_images = len(corona_df)

    split_indices = []
    curr_idx = 0
    for s in splits[:-1]:
        curr_idx += int(s * num_images)
        split_indices.append(curr_idx)

    partitions = np.split(corona_df, split_indices)
    partitions = [part.sample(frac=1) for part in partitions]

    return partitions


def filter_corona_df(corona_df,
                     coronagraph_type='C4'):
    """
    Filters image data based on coronagraph type 
    """
    if coronagraph_type == 'C3':
        filenames = corona_df['image_path'].apply(lambda x: Path(x).stem)
        corona_df = corona_df[filenames.str.contains("lasc3rdf")]
        filenames = filenames[filenames.str.contains("lasc3rdf")]

        # filter out files tagged with `cme`
        corona_df = corona_df[~filenames.str.contains("cme")]

    return corona_df


#def associate_image_to_cme(corona_df,
#                           cme_df,
#                           cme_source="DONKI",
#                           pad_days=3):
#    """
#    Find the closest CME to a coronagraph frame and associate 
#    the image with that CME, returning an array of indices to 
#    CMEs associated with respective corona frames 
#    """
#    # check that cme source is recognized. If not, raise error 
#    if cme_source not in ["DONKI", "CDAW"]:
#        raise ValueError("Error. cme_source not recognized")
#
#    indices = np.empty(len(corona_df))
#    for i, row in corona_df.iterrows():
#
#        date = pd.to_datetime(row['datetime'])
#
#        # get index of closest CME event 
#        if cme_source == "DONKI":
#            idx = (cme_df['startTime'] - date).abs().argsort()[:1]
#            cme_date = cme_df['startTime'].iloc[idx]
#        elif cme_source == "CDAW":
#            idx = (cme_df['CME_time'] - date).abs().argsort()[:1]
#            cme_date = cme_df['CME_time'].iloc[idx].iloc[0]
#
#        # Check if the date of the CME is within the provided padding 
#        # If not, flag with `-1` so that we know there is no associated
#        # CME for that frame
#        if np.abs(cme_date - date) <= timedelta(pad_days):
#            indices[i] = idx 
#        else:
#            indices[i] = -1
#
#    return indices


def associate_image_sequence_to_cme(corona_sequences, cme_df):
    """
    Associates coronagraph frames to CME using date 
    """
    # ensure that dates in CME dataframe are given as datetime objects
    try:
        cme_df['donki_date'] = cme_df['donki_date'].map(
            lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        )
    except TypeError:
        pass

    # Match CMEs to sequences of frames by date and time
    matched_frame_sequences = list()
    matched_cme_events = list()
    for sequence in corona_sequences:

        # Get the end and beginning times for the sequence
        sequence_start_time = sequence[0]['date_of_capture']
        sequence_end_time = sequence[-1]['date_of_capture']

        # find CMEs that occur within this time frame
        matched_cme_df = cme_df[((sequence_start_time <= cme_df['donki_date'])
                                &(sequence_end_time >= cme_df['donki_date']))]

        # If more than one CME was matched to the sequence, match CME that 
        # occurs closer to end of sequence (more temporal information to 
        # use to make intensity predictions)
        if len(matched_cme_df) > 1:
            idx = (sequence_end_time - matched_cme_df['donki_date']).abs().idxmin()
            matched_cme_event = pd.DataFrame(matched_cme_df.iloc[idx])
            matched_cme_events.append(matched_cme_event)
            matched_frame_sequences.append(sequence)

        elif len(matched_cme_df) == 1:
            matched_cme_event = pd.DataFrame(matched_cme_df.iloc[0])
            matched_cme_events.append(matched_cme_event)
            matched_frame_sequences.append(sequence)

    # return matched sequences and matched events
    return matched_cme_events, matched_frame_sequences


def label_single_frame_data(corona_df,
                            cme_df,
                            cme_source="DONKI",
                            pad_days=3):
    """
    Label data as sep event (1) or not (0)
    binary classification

    :param corona_df: (pandas df) dataframe containing dates and paths to corona 
     image frames
    :param ranges: [(,)] list of tuples consisting of onset and threshold times of SEP 
     events, as output by the `match_sep_events` function
    :param cme_df: (pandas df) dataframe containing data read from CME datafile
    :param cme_type: (str) source of CME data (DONKI or CDAW) 

    :return labels: array of labels for images in corona dataframe
    """
    # associate images with CME
    indices = associate_image_to_cme(corona_df, cme_df, 
            cme_source=cme_source, pad_days=pad_days)

    # mask images that have no associated CME
    mask = indices != -1
    indices = indices[mask]
    corona_df = corona_df[mask]

    # Get associated CMEs for images
    associated_cme_df = cme_df.loc[indices]

    # Get target vector from cme dataframe. This will serve 
    # as label for images
    labels = associated_cme_df['target'].to_numpy()
    corona_df['labels'] = labels

    return corona_df


def label_sequence_dir_data(
    sequence_names: List[str],
    cme_df: pd.DataFrame,
    sep_events_df: pd.DataFrame,
    elevated_proton_events_df: pd.DataFrame,
    pred_targets: List[str]=['peak_intensity']
):

    # Get CDAW date associated to sequence from directory name
    sequence_dates = list()
    for seq_name in sequence_names:
        date = datetime.strptime(seq_name, "%Y%m%d_%H%M%S")
        sequence_dates.append(date)

    # convert CME CDAW date column to datetime object
    cme_df['cdaw_date'] = cme_df['cdaw_date'].map(
        lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S") if  
        isinstance(t, str) else t
    )

    # initalize an index column for the CME dataframe that contains 
    # the index of the sequence date the CME is matched to
    cme_df['sequence_index'] = -1

    # match CME to sequence by CDAW date
    for i, seq_date in enumerate(sequence_dates):

        cme_df['sequence_index'].loc[cme_df[cme_df['cdaw_date'] == seq_date].index] = i

    # if no sequence was matched to a CME row, remove it from dataframe
    cme_df = cme_df[cme_df['sequence_index'] != -1]

    # match proton events to CMEs 
    sep_labels, sep_correlated_cme_df = correlate_cme_to_proton_event(
        sep_events_df,
        cme_df,
        target_label=1,
        pad_days=3,
    )

    # label SEP correlated CME df with target `1` so they can be distinguished 
    # in the test dataframe 
    sep_correlated_cme_df['target'] = 1

    # filter out already matched cmes
    keys = list(sep_correlated_cme_df.drop(['proton_event_start',
        'proton_event_threshold_time', 'proton_event_peak_time'], 1).columns.values)
    i1 = cme_df.set_index(keys).index
    i2 = sep_correlated_cme_df.set_index(keys).index
    cme_df = cme_df[~i1.isin(i2)]

    elevated_proton_labels, elevated_proton_correlated_cme_df = correlate_cme_to_proton_event(
        elevated_proton_events_df,
        cme_df,
        target_label=0,
        pad_days=3,
    )

    # label proton elvated event correlated CME df with target `2` so they can be distinguished 
    # in the test dataframe 
    elevated_proton_correlated_cme_df['target'] = 2

    # drop SEP correlated and non-SEP photon event correlated cme events from
    # main dataframe
    uncorrelated_cme_df = cme_df.drop(index=elevated_proton_correlated_cme_df.index)

    # apply labels to dataframes
    for target_label in pred_targets:
        if target_label in ["start_time", "threshold_time", "peak_time"]:
            sep_correlated_cme_df[target_label] = get_date_labels_by_minute(
                sep_correlated_cme_df, sep_labels, label=target_label,
            )
            elevated_proton_correlated_cme_df[target_label] = get_date_labels_by_minute(
                elevated_proton_correlated_cme_df, elevated_proton_labels, label=target_label,
            )
            uncorrelated_cme_df[target_label] = 5000.0

        elif target_label == 'peak_intensity':
            sep_proton_intensities = sep_labels[2::4]['Intensity']
            sep_proton_intensities.index = sep_correlated_cme_df.index
            sep_correlated_cme_df['peak_intensity'] = sep_proton_intensities

            elevated_proton_intensities = elevated_proton_labels[2::4]['Intensity']
            elevated_proton_intensities.index = elevated_proton_correlated_cme_df.index
            elevated_proton_correlated_cme_df['peak_intensity'] = elevated_proton_intensities

            uncorrelated_cme_df['peak_intensity'] = 10.0 / np.e**2

    return (sequence_dates, sep_correlated_cme_df,
            elevated_proton_correlated_cme_df, uncorrelated_cme_df)


def label_multiframe_data(frame_paths,
                          cme_df,
                          sep_events_df,
                          elevated_proton_events_df,
                          cme_source="DONKI",
                          frames_per_sequence=15,
                          pred_targets=["peak_intensity"],
                          pad_days=3):
    """
    Given a series of frames for a day, label if there is a CME 
    event during the day or not
    """
    # decipher time of capture from frame filename
    dates_of_capture = list()
    for path in frame_paths:
        basename = get_basename(path)
        date_string = ' '.join(basename.split('_')[:2])
        date = datetime.strptime(date_string, "%Y%m%d %H%M%S")
        dates_of_capture.append(date)

    image_data = [
        {"image_path": path, "date_of_capture": date}
        for path, date in zip(frame_paths, dates_of_capture)
    ]

    # sort data by date
    image_data.sort(key=lambda d: d['date_of_capture'])

    image_sequences = list(to_batch(image_data, frames_per_sequence))

    # link image sequences to CME events
    matched_cme_events, matched_sequences = associate_image_sequence_to_cme(image_sequences, cme_df)

    matched_cme_df = pd.concat(matched_cme_events, axis=1).T

    # add an index column for matched cme events that corresponds to the frame sequence it was 
    # matched to
    matched_cme_df['sequence_index'] = np.arange(len(matched_sequences), dtype=np.int32)

    # match proton events to CMEs 
    sep_labels, sep_correlated_cme_df = correlate_cme_to_proton_event(
        sep_events_df,
        matched_cme_df,
        target_label=1,
        pad_days=pad_days,
    )

    # label SEP correlated CME df with target `1` so they can be distinguished 
    # in the test dataframe 
    sep_correlated_cme_df['target'] = 1

    # filter out already matched cmes
    keys = list(sep_correlated_cme_df.drop(['proton_event_start',
        'proton_event_threshold_time'], 1).columns.values)
    i1 = matched_cme_df.set_index(keys).index 
    i2 = sep_correlated_cme_df.set_index(keys).index
    matched_cme_df = matched_cme_df[~i1.isin(i2)]

    elevated_proton_labels, elevated_proton_correlated_cme_df = correlate_cme_to_proton_event(
        elevated_proton_events_df,
        matched_cme_df,
        target_label=0,
        pad_days=3,
    )

    # label proton elvated event correlated CME df with target `2` so they can be distinguished 
    # in the test dataframe 
    elevated_proton_correlated_cme_df['target'] = 2

    # drop SEP correlated and non-SEP photon event correlated cme events from
    # main dataframe
    uncorrelated_cme_df = matched_cme_df.drop(index=elevated_proton_correlated_cme_df.index)

    # apply labels to dataframes
    for target_label in pred_targets:
        if target_label in ["start_time", "threshold_time", "peak_time"]:
            sep_correlated_cme_df[target_label] = get_date_labels_by_minute(
                sep_correlated_cme_df, sep_labels, label=target_label,
            )
            elevated_proton_correlated_cme_df[target_label] = get_date_labels_by_minute(
                elevated_proton_correlated_cme_df, elevated_proton_labels, label=target_label,
            )
            uncorrelated_cme_df[target_label] = 5000.0

        elif target_label == 'peak_intensity':
            sep_proton_intensities = sep_labels[2::4]['Intensity']
            sep_proton_intensities.index = sep_correlated_cme_df.index
            sep_correlated_cme_df['peak_intensity'] = sep_proton_intensities 

            elevated_proton_intensities = elevated_proton_labels[2::4]['Intensity']
            elevated_proton_intensities = elevated_proton_correlated_cme_df.index
            elevated_proton_correlated_cme_df['peak_intensity'] = elevated_proton_intensities

            uncorrelated_cme_df['peak_intensity'] = 10.0 / np.e**2

    return matched_sequences, sep_correlated_cme_df, elevated_proton_correlated_cme_df, uncorrelated_cme_df


#def label_multiframe_data(frame_paths,
#                          cme_df,
#                          sep_events_df,
#                          elevated_proton_events_df,
#                          cme_source="DONKI",
#                          frames_per_sequence=15,
#                          pred_targets=["peak_intensity"],
#                          pad_days=3):
#
#    # partition CME dataframes
#    pass
    

def serialize_sequence_data(image_data_sequence: np.array,
                            data_label: pd.DataFrame,
                            image_shape: Tuple[int] or List[int]=None,
                            make_images_grayscale: bool=True,
                            target_labels=['peak_intensity']):
    """
    Serialize sequence data into tfrecord example 
    """
    # prepare feature dictionary
    if image_shape is not None:
        image_height = image_shape[0]
        imafe_width = image_shape[1]
    else:
        image_height = 512
        image_width = 512

    if len(image_data_sequence) == 0:
        import pdb; pdb.set_trace()

    # read images 
    image_list = list()
#    for image_data in image_data_sequence:
    for image_path in image_data_sequence:
#        image_path = image_data['image_path']
        image = cv2.imread(image_path)
        if make_images_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.uint8)
        image_list.append(image)

    image_stack = np.stack(image_list, axis=0)

    # Get the date of this sequence from directory path
    sequence_date = os.path.basename(os.path.dirname(image_data_sequence[0]))

    feature = {
        'height': _int64_feature(image_height),
        'width': _int64_feature(image_width),
        'channels': _int64_feature(1 if make_images_grayscale else 3),
        'class_id': _int64_feature(data_label['target'] + 1),
        'num_frames': _int64_feature(len(image_list)),
        'sequence_raw': _bytes_feature(image_stack.tostring()),
        'sequence_date': _bytes_feature(bytes(sequence_date, 'utf-8')),
    }

    for label in target_labels:
        feature[label] = _float_feature(data_label[label])

    return tf.train.Example(features=tf.train.Features(feature=feature))


def serialize_data(image_path, 
                   image_shape,
                   label):
    """
    Serialize image data into tfrecord example
    """
    height, width, channels = image_shape
    image_string = open(image_path, 'rb').read()
    feature = {
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'channels': _int64_feature(channels),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string)
    } 

    return tf.train.Example(feature=tf.train.Features(feature=feature))


def write_tfrecords_by_sequence_dir(
        image_data_path: str,
        output_path: str,
        cme_data_path: str,
        ten_mev_event_datafile: str,
        ten_mev_re2_event_datafile: str,
        train_partition: float=0.5,
        valid_partition: float=0.2,
        test_partition: float=0.3,
        num_examples_per_tfrecord: int=8,
):
    """
    Writes TF Records if the frames are already assigned to their corresponding CME 
    by CDAW date (in this case, the date of the corresponding CDAW data will be the 
    directory name for the sequence 
    """
    # ask if it's okay to remove output path
    if os.path.exists(output_path):
        if ask_for_confirmation(f"Warning: removing contents in {output_path}. Continue?"):
            shutil.rmtree(output_path)

    os.makedirs(output_path)

    train_path = os.path.join(output_path, "train")
    valid_path = os.path.join(output_path, "valid")
    test_path = os.path.join(output_path, "test")

    os.mkdir(train_path)
    os.mkdir(valid_path)
    os.mkdir(test_path)

    # Get all sub-directories in the root path
    def _get_immediate_subdirectories(root_dir):
        return [name for name in os.listdir(root_dir) if
                os.path.isdir(os.path.join(root_dir, name))]

    sequence_directories = _get_immediate_subdirectories(image_data_path)

    # Parse function to check that we can convery a directory name to desired format
    def _date_parse_check(dir_name):
        try:
            datetime.strptime(dir_name, "%Y%m%d_%H%M%S")
            return True
        except ValueError:
            return False


    # ensure that all sequence directories can be interpreted as dates
    sequence_directories = list(filter(_date_parse_check, sequence_directories))

    # Check to see if all sequence directories actually have frames in them. If not,
    # remove
    sequence_directories = list(filter(lambda d: len(glob.glob(os.path.join(
        image_data_path, d, "*.png"))) > 0, sequence_directories))

    # load cme dataframe 
    cme_df = pd.read_csv(cme_data_path)

    # drop repeat CDAW dates
    cme_df.drop_duplicates(subset=['cdaw_date'], keep='first', inplace=True)

    # Get dataframes for proton events and split into SEP and elevated proton event
    # dataframes
    sep_events_df, elevated_proton_events_df = prepare_10MeV_proton_event_dataframes(
        ten_mev_event_datafile, ten_mev_re2_event_datafile
    )

    # Label all sequences with proton intensity (when available)
    (multi_frame_sequence, sep_correlated_cme_df,
            elevated_proton_correlated_cme_df, uncorrelated_cme_df) = label_sequence_dir_data(
        sequence_directories,
        cme_df,
        sep_events_df,
        elevated_proton_events_df,
        pred_targets=TARGET_LABELS,
    )

    # divide data into train, valid, and test partitions
    train_cme_df, valid_cme_df, test_cme_df = partition_dataset(
        sep_correlated_cme_df,
        elevated_proton_correlated_cme_df,
        uncorrelated_cme_df,
        partition_method='sep-and-non-sep-balanced',
        train_partition=train_partition,
        valid_partition=valid_partition,
        test_partition=test_partition,
    )

    multi_frame_sequence_array = np.array(multi_frame_sequence)
    train_frame_sequence_dirs = multi_frame_sequence_array[train_cme_df['sequence_index']
            .to_numpy(np.int32)].tolist()
    valid_frame_sequence_dirs = multi_frame_sequence_array[valid_cme_df['sequence_index']
            .to_numpy(np.int32)].tolist()
    test_frame_sequence_dirs = multi_frame_sequence_array[test_cme_df['sequence_index']
            .to_numpy(np.int32)].tolist()

    # convert the datetime object sequence dirs back into strings
    train_frame_sequence_dirs = list(map(lambda d: datetime.strftime(d, '%Y%m%d_%H%M%S'),
                                     train_frame_sequence_dirs))
    valid_frame_sequence_dirs = list(map(lambda d: datetime.strftime(d, '%Y%m%d_%H%M%S'),
                                     valid_frame_sequence_dirs))
    test_frame_sequence_dirs = list(map(lambda d: datetime.strftime(d, '%Y%m%d_%H%M%S'),
                                     test_frame_sequence_dirs))

    # Get paths to frames from sequence subdirectories
    train_frame_sequence_paths = list()
    for seq_dir in train_frame_sequence_dirs:
        sequence = glob.glob(os.path.join(image_data_path, seq_dir, "*.png"))
        train_frame_sequence_paths.append(sequence)

    valid_frame_sequence_paths = list()
    for seq_dir in valid_frame_sequence_dirs:
        sequence = glob.glob(os.path.join(image_data_path, seq_dir, "*.png"))
        valid_frame_sequence_paths.append(sequence)

    test_frame_sequence_paths = list()
    for seq_dir in test_frame_sequence_dirs:
        sequence = glob.glob(os.path.join(image_data_path, seq_dir, "*.png"))
        test_frame_sequence_paths.append(sequence)

    # write partitions to disk
    write_partition_to_tfrecord(
        train_cme_df,
        train_frame_sequence_paths,
        "train",
        train_path,
        target_labels=TARGET_LABELS,
        num_examples_per_tfrecord=num_examples_per_tfrecord,
    )

    write_partition_to_tfrecord(
        valid_cme_df,
        valid_frame_sequence_paths,
        "valid",
        valid_path,
        target_labels=TARGET_LABELS,
        num_examples_per_tfrecord=num_examples_per_tfrecord,
    )

    write_partition_to_tfrecord(
        test_cme_df,
        test_frame_sequence_paths,
        "test",
        test_path,
        target_labels=TARGET_LABELS,
        num_examples_per_tfrecord=num_examples_per_tfrecord,
    )


def write_tfrecords_by_n_frame_association():
    """
    Writes Tf Records if frames have not already been assigned to their corresponding 
    CME. In this case, take the 'n' frames that precede the CME date as the associated 
    sequence 
    """
    # generate tfrecords
    output_path = flags.output_path
    
    # ask if it's okay to remove output path
    if os.path.exists(output_path):
        if ask_for_confirmation(f"Warning: removing contents in {output_path}. Continue?"):
            shutil.rmtree(output_path)

    os.makedirs(output_path)

    train_path = os.path.join(output_path, "train")
    valid_path = os.path.join(output_path, "valid")
    test_path = os.path.join(output_path, "test")

    os.mkdir(train_path)
    os.mkdir(valid_path)
    os.mkdir(test_path)

#    image_shape = (512, 512, 1)
#    num_images_per_record = flags.num_images_per_record

    # get all image frames in datapath
    frame_paths = glob.glob(os.path.join(flags.image_data_path, "*.png"))

    # load cme dataframe
    cme_df = pd.read_csv(flags.cme_data_path)

    # Get dataframes for proton events and split into SEP and elevated proton event
    # dataframes
    sep_events_df, elevated_proton_events_df = prepare_10MeV_proton_event_dataframes(
        flags.ten_mev_event_datafile, flags.ten_mev_re2_event_datafile
    )

    # get labels for data
    if flags.associate_cme_to_n_frames:
        (multi_frame_sequences, sep_correlated_cme_df, 
                elevated_proton_correlated_cme_df, uncorrelated_cme_df) = label_multiframe_data(
            frame_paths,
            cme_df,
            sep_events_df,
            elevated_proton_events_df,
            cme_source=flags.cme_source,
            frames_per_sequence=flags.num_frames_per_sequence,
            pred_targets=["peak_intensity"],
            pad_days=flags.pad_days,
        )
    elif flags.associate_cme_to_sequence_directory:
        (multi_frame_sequence, sep_correlated_cme_df,
                elevated_proton_correlated_cme_df, uncorrelated_cme_df) = label_multiframe_data(
            frame_paths,
            cme_df,
            sep_events_df,
            elevated_proton_events_df,
            cme_source=flags.cme_source,
            pred_targets=["peak_intensity"],
        )

    # save sequences to disk with associated DONKI timestamp
    if flags.save_sequences_to_disk:
        sequences_dir = os.path.join(output_path, "saved_sequences")
        cme_df = pd.concat([sep_correlated_cme_df, elevated_proton_correlated_cme_df,
                            uncorrelated_cme_df])
        for cme_event in cme_df.iterrows():
            correlated_sequence = multi_frame_sequences[cme_event['sequence_index']]
            write_sequence_to_disk(correlated_sequence, cme_event['donki_date'],
                                   sequences_dir)

    # divide data into train, valid, and test partitions
    train_cme_df, valid_cme_df, test_cme_df = partition_dataset(
        sep_correlated_cme_df,
        elevated_proton_correlated_cme_df,
        uncorrelated_cme_df,
        train_partition=flags.train_partition,
        valid_partition=flags.valid_partition,
        test_partition=flags.test_partition,
    )

    # NOTE: remove when full dataset becomes available...
    # move test SEP events to train as well
    # just for testing purposes
    train_cme_df = train_cme_df.append(test_cme_df[test_cme_df['peak_intensity'] > 10.0 / np.e**2])

    # divide multi frame sequences into appropriate partition
    multi_frame_sequence_array = np.array(multi_frame_sequences)
    train_frame_sequences = multi_frame_sequence_array[train_cme_df['sequence_index'].to_numpy(np.int32)]
    valid_frame_sequences = multi_frame_sequence_array[valid_cme_df['sequence_index'].to_numpy(np.int32)]
    test_frame_sequences = multi_frame_sequence_array[test_cme_df['sequence_index'].to_numpy(np.int32)]

    # write partitions to disk
    write_partition_to_tfrecord(
        train_cme_df,
        train_frame_sequences,
        "train",
        train_path,
        num_examples_per_tfrecord=flags.num_examples_per_tfrecord,
    )

    write_partition_to_tfrecord(
        valid_cme_df,
        valid_frame_sequences,
        "valid",
        valid_path,
        num_examples_per_tfrecord=flags.num_examples_per_tfrecord,
    )

    write_partition_to_tfrecord(
        test_cme_df,
        test_frame_sequences,
        "test",
        test_path,
        num_examples_per_tfrecord=flags.num_examples_per_tfrecord,
    )


def main(flags):
    if flags.write_tfrecords_by_sequence_dir:
        write_tfrecords_by_sequence_dir(
            image_data_path=flags.image_data_path,
            output_path=flags.output_path,
            cme_data_path=flags.cme_data_path,
            ten_mev_event_datafile=flags.ten_mev_event_datafile,
            ten_mev_re2_event_datafile=flags.ten_mev_re2_event_datafile,
            train_partition=flags.train_partition,
            valid_partition=flags.valid_partition,
            test_partition=flags.test_partition,
            num_examples_per_tfrecord=flags.num_examples_per_tfrecord,
        )

    elif flags.write_tfrecords_by_n_frame_match:
        write_tfrecords_by_n_frame_association()
    else:
        raise NotImplementedError("Method for generating Tensorflow Records not recognized")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_data_path', type=str,
                        required=True,
                        help="Datapath for SOHO image data")

    parser.add_argument('--cme_data_path', type=str,
                        required=True,
                        help="Path for CME data")

    parser.add_argument('--cme_source', type=str,
                        default="DONKI-extended",
                        help="Source for CME data. Possible options are:\n"
                        + "\t1) DONKI\n"
                        + "\t2) CDAW")

    parser.add_argument('--ten_mev_event_datafile', type=str,
                        help="Path for 10 MeV datafile")

    parser.add_argument('--ten_mev_re2_event_datafile', type=str,
                        help="Path for 10 MeV re2 datafile")

#    parser.add_argument('--corona_type', type=str,
#                        required=True,
#                        default='C3',
#                        help="Coronagraph type to collect data for."
#                        + "Options are: \n"
#                        + "\t1) C2\n"
#                        + "\t2) C3\n")

    parser.add_argument('--train_partition', type=float,
                        default=0.5,
                        help="Fraction of data to use for training partition")

    parser.add_argument('--valid_partition', type=float,
                        default=0.2,
                        help="Fraction of data to use for validation partition")

    parser.add_argument('--test_partition', type=float,
                        default=0.3,
                        help="Fraction of data to use for test partition")

    parser.add_argument('--pad_days', type=int,
                        default=3,
                        help="Number of days of padding to allow when associating CME to image frame")

    parser.add_argument('--num_frames_per_sequence', type=int,
                        default=15,
                        help="Number of frames to use per coronagraph sequence")

    parser.add_argument('--output_path', type=str,
                        required=True,
                        help="Output path for serialized tfrecords")
    
    parser.add_argument('--num_examples_per_tfrecord', type=int,
                        default=8,
                        help="Number of images to include in each tfrecord")

    # mutually exclusive arguments governing how to generate tfrecords 
    gen_method_group = parser.add_mutually_exclusive_group(required=True)

    gen_method_group.add_argument('--write_tfrecords_by_sequence_dir', 
                                  action='store_true',
                                  default=False,
                                  help="Flag to set to write TF Records for data that was collected "
                                  + "already assigned to a CME event by directory name, where the "
                                  + "directory name is the CDAW date of the associated CME")

    gen_method_group.add_argument('--write_tfrecords_by_n_frame_association',
                                  action='store_true',
                                  default=False,
                                  help="Write Tensorflow Records by association the n-frames that "
                                  + "occur before a CME's CDAW date to that CME")

    flags = parser.parse_args()

    main(flags)
