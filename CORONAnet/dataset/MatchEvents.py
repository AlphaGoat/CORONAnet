import argparse
import csv
import pandas as pd
import numpy as np

def match_image_to_cme(filename,
                       cme_df):
    """
    Given a filename, return index of cme 
    in dataframe that provides closest date 
    match 
    """
    pass

def match_images_to_cmes(image_df,
                         cdaw_df):
    """
    Matches images to CMEs closest to it in date and returns 
    dataframe with image path, date, and index of correlated CME 
    """

    correlated_indices = []
    for _, row in image_df.iterrows():
        i = np.argmin(np.abs(cme_df['CME_time'] - image_df['Image_time'])) 
        correlated_indices.append(i)

    return correlated_indices 


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_csv', type=str, 
                        help="Path to image csv")

    parser.add_argument('--cdaw_csv', type=str,
                        help="Path to CDAW CME data csv")

    flags = parser.parse_args()

    image_df = pd.read_csv(flags.image_csv)
    image_df.columns = ['Image_time', 'filepath']
    cme_df = pd.read_csv(flags.cdaw_csv)

    correlated_indices = match_images_to_cmes(image_df, cme_df)

    image_df['CME_indices'] = correlated_indices
