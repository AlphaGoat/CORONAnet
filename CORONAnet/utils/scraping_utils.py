"""
File of scraping utilities to pull CDAW C2
coronagraph videos

author: Peter Thomas
date: September 03, 2021
"""
import os
import re
import csv
import glob
import time 
import shutil
import random
import logging
import argparse
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
from posixpath import basename, dirname

from CORONAnet.utils import ask_for_confirmation, get_basename


def download_file(url, datapath):

    local_filename = os.path.join(datapath, url.split('/')[-1])

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
                desc=url,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=8192
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

    return local_filename


def scrape_lasco_c2_frames(cme_df, download_path, remove_existing=False):
    """
    Scrape LASCO C2 frames corresponding to CDAWs in CME dataframe 
    """
    # Get column of CDAW dates and times  
    cme_df['cdaw_date'] = cme_df['cdaw_date'].map(
        lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S") 
        if isinstance(t, str) else t
    )
    cdaw_dates = cme_df['cdaw_date']

    # see which CDAW events have already been downloaded in CDAW folder 
    # (if we are not replacing all events)
    if not remove_existing:
        if os.path.exists(download_path):
            subfolders = [os.path.basename(f.path) for f in os.scandir(download_path) if f.is_dir()]
            downloaded_cdaw_dates = [datetime.strptime(f, "%Y%m%d_%H%M%S") for f in subfolders]
            downloaded_cdaw_dates.sort()

            # Remove the last downloaded CDAW date (may not have been fully downloaded when the last
            # job was killed
            downloaded_cdaw_dates = downloaded_cdaw_dates[:-1] 

            cdaw_dates = pd.Series([c for c in cdaw_dates if c not in downloaded_cdaw_dates])
    else:
        if os.path.exists(download_path):
            shutil.rmtree(download_path)

    base_url = "https://cdaw.gsfc.nasa.gov"
    cme_list_url = base_url + "/CME_list/UNIVERSAL/"

    # iterate through cdaw dates and retrieve sequences of images 
    for i, date in enumerate(cdaw_dates):
        # Get the html for the CDAW page for the month 
        month_url = cme_list_url + str(date.year) + f"_{date.month:02d}"
        month_url += "/univ" + str(date.year) + f"_{date.month:02d}.html"
        while True:
            try:
                month_res = requests.get(month_url)
                break
            except requests.exceptions.ConnectionError:
                print("Connection error. Sleeping it off...")
                time.sleep(1.0)

        month_soup = BeautifulSoup(month_res.text, 'html.parser')

        # parse html for month page to get the details for this specific date
        index_of_cdaw = -1
        hd_date_time_elements = month_soup.find_all('td', headers='hd_date_time')
        for j, element in enumerate(hd_date_time_elements):
            if j % 2 == 1:
                continue
            if (datetime.strptime(element.text.replace('\n', '').replace(' ', ''),
                    "%Y/%m/%d").day - date.day) == 0:
                if (datetime.strptime(hd_date_time_elements[j + 1].text.replace('\n', '').replace(
                    ' ', ''), "%H:%M:%S").time().replace(second=0) == date.time()):
                    # Get url for java movie
                    jsmovie_url = os.path.join(os.path.dirname(month_url), 
                                               element.find_all('a', href=True)[0]['href'])

                    # make requests to script that calls movie url
                    jsmovie_res = requests.get(jsmovie_url)
                    if not jsmovie_res:
                        continue
                    jsmovie_soup = BeautifulSoup(jsmovie_res.text, 'html.parser')
                
                    # get movie url  
                    content = jsmovie_soup.find('meta').attrs['content'] 
                    movie_url = content[content.find('URL=')+4:]
                    movie_url = base_url + movie_url

                    # finally, make request to movie url and get the number of frames 
                    # to include in sequence as well as the start and end time for the 
                    # sequence
                    movie_re = requests.get(movie_url)
                    movie_soup = BeautifulSoup(movie_re.text, 'html.parser')
                    time.sleep(random.randint(1, 5))

                    # iterate over frames in the java script portion of the html page
                    frame_links = list()
                    jscript_segments = movie_soup.find('script').string.split('\n')
                    for jssegment in jscript_segments:
                        if jssegment.find('jfiles1.push') != -1:
                            frame_url = jssegment.replace('jfiles1.push(', '')
                            frame_url = frame_url.replace(');', '')
                            frame_url = frame_url.replace('"', '')
                            frame_url = frame_url.replace(' ', '')
                            frame_links.append(frame_url)

                    # Now that we have all of the links to frames, download imagery 
                    # to a folder specifically for this CDAW event and create a movie 
                    # sequence from the frames
                    sequence_savedir = os.path.join(download_path, 
                                                    date.strftime('%Y%m%d_%H%M%S'))
                    os.makedirs(sequence_savedir, exist_ok=True)
                    frame_paths = list()
                    for link in frame_links:
                        try:
                            save_path = download_file(link, sequence_savedir)
                            frame_paths.append(save_path)
                        except requests.exceptions.ConnectionError:
                            logging.info(f"unable to download image file at {link}")
                        # wait for some period of time before making next request
                        finally:
                            print("sleeping...")
                            time.sleep(random.randint(1, 5))

                    # once all frames in sequence have been downloaded, order all 
                    # frames in sequence and output a movie of the sequence 

                    # first, get date from filename (NOTE: all files should already 
                    # be ordered in sequence, this is just a sanity check)
                    basenames = ['_'.join(get_basename(path).split('_')[:2]) for path in frame_paths]
                    datetimes_of_files = np.array([datetime.strptime(datetime_str, '%Y%m%d_%H%M%S') 
                                                  for datetime_str in basenames])
                    sorted_indices = np.argsort(datetimes_of_files)

                    sorted_frame_paths = [frame_paths[idx] for idx in sorted_indices]

                    if sorted_frame_paths:

                        frames = list()
                        for path in sorted_frame_paths:
                            new_image = Image.open(path)
                            frames.append(new_image)

                        movie_dir = os.path.join(sequence_savedir, "movies")
                        os.makedirs(movie_dir, exist_ok=True)
                        frames[0].save(os.path.join(movie_dir, 'movie.gif'), format='GIF', 
                                       append_images=frames[1:], save_all=True,
                                       duration=300, loop=0)

        print("sleeping...")
        time.sleep(random.randint(1, 5))


def resume_lasco_scraping(datapath):
    """
    Resumes scraping LASCO dataset from where we left off 
    """
    base_image_url = "https://cdaw.gsfc.nasa.gov/images/soho/lasco/"

    # get text file with image urls
    textfile_save_dir = os.path.join(datapath, "text_files")
    text_paths = glob.glob(os.path.join(textfile_save_dir, "*.txt"))

    c2_savepath = os.path.join(datapath, "C2_imagery")

    # get all images that have already been downloaded
    saved_frames = glob.glob(os.path.join(c2_savepath, "*.png"))
    saved_frames = [os.path.basename(frame_path) for frame_path in saved_frames]

    c2_datafile = None
    for textfile in text_paths:
        if textfile.endswith('c2_all.txt'):
            c2_datafile = textfile

    if c2_datafile is not None:
        with open(c2_datafile, 'r') as f:
            c2_image_link_suffix = f.readlines()[1:]

        for i, suffix in enumerate(c2_image_link_suffix):
            suffix = suffix.replace('\n', '')
            if int(suffix.split('/')[0]) < 2010:
                continue
            elif os.path.basename(suffix) in saved_frames:
                continue

            c2_image_link = os.path.join(base_image_url, suffix)

            try:
                savepath = download_file(c2_image_link, c2_savepath)
            except requests.exceptions.ConnectionError:
                print(f"unable to download file at {c2_image_link}")

            # wait for specified amount of time before making next request
            finally:
                time.sleep(random.randint(1,5)) 


def scrape_lasco_imagery(datapath):
    """
    Scrape all LASCO imagery 
    """
    # create datapath
    if os.path.exists(datapath): 
        if ask_for_confirmation(f"Directory at {datapath} will be removed. Is this okay?"):
            shutil.rmtree(datapath)
        else:
            print("Halting scraping operations....")
            return

    os.mkdir(datapath)

    # grab text files detailing datafiles for each year of LASCO/SOHO collections
    text_url = "https://cdaw.gsfc.nasa.gov/images/soho/lasco/filelist/"
    res = requests.get(text_url)
    soup = BeautifulSoup(res.text, 'html.parser')

    # make a directory to store text files
    textfile_save_dir = os.path.join(datapath, "text_files")
    os.mkdir(textfile_save_dir)

    # download all text files 
    text_paths = list()
    txt_elements = soup.find_all('a', href=True)
    for element in txt_elements:
        # check if element has a text file link
        link = element['href']
        if not link.endswith('_all.txt'):
            continue
        else:
            savepath = download_file(os.path.join(text_url, link), textfile_save_dir)
            text_paths.append(savepath)

            # wait for specified amount of time before making next request
            time.sleep(random.randint(1,5)) 

    base_image_url = "https://cdaw.gsfc.nasa.gov/images/soho/lasco/"

    # open c2 data text file, get links to imagery, and download to datapath
    c2_savepath = os.path.join(datapath, "C2_imagery")
    os.mkdir(c2_savepath)

    c2_datafile = None
    for textfile in text_paths:
        if textfile.endswith('c2_all.txt'):
            c2_datafile = textfile

    if c2_datafile is not None:
        with open(c2_datafile, 'r') as f:
            c2_image_link_suffix = f.readlines()[1:]

        for suffix in c2_image_link_suffix:
            if int(suffix.split('/')[0]) < 2010:
                continue

            c2_image_link = os.path.join(base_image_url, suffix.replace('\n', ''))

            try:
                savepath = download_file(c2_image_link, c2_savepath)
            except requests.exceptions.ConnectionError:
                print(f"unable to download file at {c2_image_link}")

            # wait for specified amount of time before making next request
            finally:
                time.sleep(random.randint(1,5)) 

    c3_savepath = os.path.join(datapath, "C3_imagery")
    os.mkdir(c3_savepath)

    c3_datafile = None 
    for textfile in text_paths:
        if textfile.endswith('c3_all.txt'):
            c3_datafile = textfile

    if c3_datafile is not None:
        with open(c3_datafile, 'r') as f:
            c3_image_link_suffix = f.readlines()[1:]

        for suffix in c3_image_link_suffix:
            if int(suffix.split('/')[0]) < 2010:
                continue

            c3_image_link = os.path.join(base_image_url, suffix.replace('\n', ''))

            try:
                savepath = download_file(c3_image_link, c3_savepath)
            except requests.exceptions.ConnectionError:
                print(f"unable to download file at {c3_image_link}")

            # wait for specified amount of time before making next request
            finally:
                time.sleep(random.randint(1,5)) 


def scrape_sep_event_images(datapath,
                            cme_df,
                            cme_source="CDAW",
                            pad_days=1):
    """
    Grab images specifically from SEP events, as well as all imagery 
    up to pad number of days in either direction
    """
    base_url = "https://cdaw.gsfc.nasa.gov/images/soho/lasco"

    # filter out non-SEP events
    cme_df = cme_df[cme_df['target'] == 1]

    # Get the dates of SEP events and iterate over them 
    # to get imagery
    if cme_source == "CDAW":
        sep_dates = cme_df['CME_time'].map(
            lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S"))
    else:
        sep_dates = cme_df['startTime'].map(
            lambda t: datetime.strptime(t, "%Y-%m-%d %H:%M:%S"))

    # create csv to write date of CME event for each frame so 
    # that we can correlate with CME events later 
    csv_filename = os.path.join(datapath, "CDAW_image_files.csv")
    if os.path.exists(csv_filename): os.remove(csv_filename)

    for date in sep_dates:

        year = date.year 
        month = date.month 
        day = date.day

        imagedir_url = os.path.join(base_url, str(year), "{m:02d}".format(m=month))
        while True:
            res = requests.get(imagedir_url)
            if res.status_code == 200:
                break
            else:
                time.sleep(random.randint(1, 5))

        soup = BeautifulSoup(res.text, 'html.parser')
        day_elements = soup.find_all('a', href=True)
        for element in day_elements:
            # check if the element is one of the day directores
            reference_string = element['href'][:-1]
            try:
                day_ref = int(reference_string)
            except ValueError:
                continue

            # if the day is not in range given by padding, move on 
            if day_ref not in range(day - pad_days, day + pad_days):
                continue

            day_data_url = os.path.join(imagedir_url, "{day:02d}".format(day=day_ref))

            while True:
                res = requests.get(day_data_url)
                if res.status_code == 200:
                    break 
                else:
                    time.sleep(random.randint(1,5))

            soup = BeautifulSoup(res.text, 'html.parser')
            image_tags = soup.find_all('a', href=True)
            for tag in image_tags:
                data = tag['href']
                if os.path.splitext(data)[1] != '.png':
                    continue
                else:
                    image_path = os.path.join(day_data_url, data)
                    savepath = download_file(image_path, datapath)

                    with open(csv_filename, 'a') as f:
                        date = datetime(year=year, month=month, day=day)
                        writer = csv.writer(f)
                        writer.writerow([date.strftime('%Y-%m-%d'), savepath])

                    time.sleep(random.randint(1,5)) 


def scrape_soho_image_dir(datapath):
    """
    Scrapes all images in soho image directory for 
    every day in database
    """
    base_url = "https://cdaw.gsfc.nasa.gov/images/soho/lasco"
    yr_range = range(1996, 2022)
    month_range = range(2, 13)
    day_range = range(1, 32)

    # create datapath 
    if os.path.exists(datapath): 
        if ask_for_confirmation(f"Directory at {datapath} will be removed. Is this okay?"):
            shutil.rmtree(datapath)
        else:
            print("Halting scraping operations....")
            return

    os.makedirs(datapath)

    # create csv to write date of CME event for each frame 
    # so that we can correlate with CME events later
    csv_filename = os.path.join(datapath, "CDAW_image_files.csv")
    if os.path.exists(csv_filename): os.remove(csv_filename)

    for yr in yr_range:
        for month in month_range:
            imagedir_url = os.path.join(base_url, str(yr), "{m:02d}".format(m=month))
            while True:
                res = requests.get(imagedir_url)
                if res.status_code == 200:
                    break
                else:
                    time.sleep(random.randint(1,5))

            soup = BeautifulSoup(res.text, 'html.parser')
            day_elements = soup.find_all('a', href=True)
            for el in day_elements:
                # check if the element is one of the day directories
                ref_str = el['href'][:-1]
                try:
                    day = int(ref_str)
                except ValueError:
                    continue

                day_data_url = os.path.join(imagedir_url, "{day:02d}".format(day=day))
                while True:
                    res = requests.get(day_data_url)
                    if res.status_code == 200:
                        break 
                    else:
                        time.sleep(random.randint(1,5))

                soup = BeautifulSoup(res.text, 'html.parser')
                image_tags = soup.find_all('a', href=True)
                for tag in image_tags:
                    data = tag['href']
                    if os.path.splitext(data)[1] != '.png':
                        continue
                    else:
                        image_path = os.path.join(day_data_url, data)
                        savepath = download_file(image_path, datapath)

                        with open(csv_filename, 'a') as f:
                            date = datetime(year=yr, month=month, day=day)
                            writer = csv.writer(f)
                            writer.writerow([date.strftime('%Y-%m-%d'), savepath])

                        time.sleep(random.randint(1,5)) 


def scrape_C3_data(datapath):
    """
    scrape C3 coronagraph frames from CDAW 
    catalog
    """
    base_url = "https://cdaw.gsfc.nasa.gov/CME_list/"
    yr_range = range(1996, 2022)
    month_range = range(1, 13)

    # create datapath (and ask if you want to remove dataset, if it is already present)
    if os.path.exists(datapath): 
        if ask_for_confirmation(f"Directory at {datapath} will be removed. Is this okay?"):
            shutil.rmtree(datapath)
        else:
            print("Halting data scraping operations...")

    os.makedirs(datapath)

    # create csv to write date of CME event for each frame 
    # so that we can correlate with CME events later
    csv_filename = os.path.join(datapath, "CDAW_C3_files.csv")
    if os.path.exists(csv_filename): os.remove(csv_filename)

    for yr in yr_range:
        for month in month_range:
            table_url = os.path.join(base_url, "UNIVERSAL/{yr}_{m:02d}/univ{yr}_{m:02d}.html".format(
                    yr=yr, m=month))
            base_data_url = os.path.join(base_url, "nrl_mpg")
            while True:
                res = requests.get(table_url)
                if res.status_code == 200:
                    break

            soup = BeautifulSoup(res.text, 'html.parser')
            video_tags = soup.find_all('td', headers='hd_link')
            for tag in video_tags:
                video_downloaded = False
                all_a = tag.find_all('a', href=True)
                for el in all_a:
                    dataurl = el['href']
                    if os.path.splitext(dataurl)[1] != '.mpg':
                        continue
                    parse_object = urlparse(dataurl)
                    file_path = basename(parse_object.path)
                    file_base = os.path.splitext(file_path)[0]
                    date_tag, video_type = file_base.split('_')
                    if video_type != 'c3':
                        continue 
                    else:
                        video_path = os.path.join(base_data_url,
                                "{yr}_{m:02d}".format(yr=yr, m=month),
                                file_path)
                        local_filename = download_file(video_path, datapath)
                        video_downloaded = True

                        with open(csv_filename, 'a') as f:
                            day = int(date_tag[-2:])
                            date = datetime(year=yr, month=month, day=day)
                            writer = csv.writer(f)
                            writer.writerow([date.strftime('%Y-%m-%d'), local_filename])

                if video_downloaded:
                   time.sleep(random.randint(1,10)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str,
                        default=None,
                        help="Base directory to save C2 videos")

    parser.add_argument('--cme_datafile', type=str,
                        default=None,
                        help="Path to CME datafile")

    run_mode_group = parser.add_mutually_exclusive_group(required=True)

    run_mode_group.add_argument('--scrape_all', action='store_true',
                                help="Scrape all images from CDAW website")

    run_mode_group.add_argument('--scrape_sep', action='store_true',
                                help="Scrape all images associated with SEPs")
    
    run_mode_group.add_argument('--scrape_lasco_c2', action='store_true',
                                help="Scape all images from LASCO SOHO C2 coronograph")

    parser.add_argument('--cme_source', type=str,
                        help="Source for CME data (CDAW or DONKI)")

    flags = parser.parse_args()

    if flags.scrape_all:
        scrape_C3_data(flags.datapath)
    elif flags.scrape_sep:
        cme_df = pd.read_csv(flags.cme_datapath)
        scrape_sep_event_images(flags.datapath,
                                cme_df,
                                flags.cme_source)
    elif flags.scrape_lasco_c2:
        # load cme datafile
        cme_df = pd.read_csv(flags.cme_datafile)
        scrape_lasco_c2_frames(cme_df, flags.datapath, remove_existing=False)
#        scrape_lasco_imagery(flags.datapath)
#        resume_lasco_scraping(flags.datapath)
#    scrape_soho_image_dir(flags.datapath)
