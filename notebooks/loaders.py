import glob
import itertools
import json
import os.path
import cv2
import pandas as pd
import requests
import tqdm
from pathlib import Path

import cleanup

data_dir = '../youtube_data'

def load_csv(filename):
    path = os.path.join(data_dir, filename)
    return pd.read_csv(path, sep=';', encoding='macroman', dtype={'category_id': 'category'})

def load_categories():
    gb_path = os.path.join(data_dir, 'GB_category_id.json')
    us_path = os.path.join(data_dir, 'US_category_id.json')
    categories = {}
    for path in [gb_path, us_path]:
        with open(path) as f:
            data = json.load(f)
        for category in data['items']:
            categories[category['id']] = category['snippet']['title']
    return categories

def load_gb_videos():
    return load_csv('GB_videos_5p.csv')

def load_us_videos():
    return load_csv('US_videos_5p.csv')

def load_and_clean_up_videos():
    vids = cleanup.merge(load_gb_videos(), load_us_videos())
    vids = cleanup.fill_missing_video_ids(vids)
    vids = cleanup.remove_duplicates(vids)
    vids = cleanup.trim_column_names(vids)
    return vids

def load_all_videos():
    path = os.path.join(data_dir, 'all_videos.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    
    vids = load_and_clean_up_videos()
    vids.to_csv(path)
    return vids

def load_features():
    path = os.path.join(data_dir, 'features.csv')
    features = pd.read_csv(path)
    features = features.drop(columns=['Unnamed: 0'])
    features = features.astype({ 'category_id': 'category' })
    return features

def load_selected_attributes():
    columns = ['category_id', 'description__ant', 'description__avail',
               'description__deal',
               'description__follow',
               'description__game',
               'description__imag',
               'description__key',
               'description__life',
               'description__music',
               'description__new',
               'description__news',
               'description__nhttp',
               'description__noggin',
               'description__show',
               'description__tag',
               'description__time',
               'description__watch',
               'description__world',
               'description__youtub',
               'description_num_chars',
               'description_num_exclamation_marks',
               'description_num_numbers',
               'description_num_periods',
               'description_num_question_marks',
               'description_num_words',
               'description_sentiment',
               'faces_fear',
               'faces_happy',
               'faces_surprise',
               'faces_total',
               'has_image',
               'l1',
               'like_ratio',
               'publish_hour',
               'publish_weekday',
               's2',
               'tag__animation',
               'tag__basketball',
               'tag__beauty',
               'tag__comedy',
               'tag__education',
               'tag__educational',
               'tag__fashion',
               'tag__food',
               'tag__funny',
               'tag__highlights',
               'tag__how to',
               'tag__humor',
               'tag__movie',
               'tag__music',
               'tag__nba',
               'tag__pop',
               'tag__rap',
               'tag__science',
               'tag__talk show',
               'tag__tech',
               'tag__technology',
               'tag__trailer',
               'tag__tutorial',
               'title__1',
               'title__2017',
               'title__2018',
               'title__christma',
               'title__day',
               'title__de',
               'title__first',
               'title__hd',
               'title__iphon',
               'title__made',
               'title__new',
               'title__offici',
               'title__play',
               'title__shoot',
               'title__tech',
               'title__trailer',
               'title__video',
               'title__world',
               'title_num_exclamation_marks',
               'title_num_numbers',
               'title_num_periods',
               'title_num_question_marks',
               'title_num_uppercase_words']

    features = load_features()
    features.describe()
    return features[columns]

def load_semi_supervised():
    path = os.path.join(data_dir, 'semi_supervised.csv')
    features = pd.read_csv(path)
    features = features.drop(columns=['Unnamed: 0'])
    return features

def load_broken_ids():
    path = os.path.join(data_dir, 'broken_ids.txt')
    broken_ids = set()
    try:
        with open(path) as f:
            for line in f:
                broken_ids.add(line.strip())
    except IOError:
        pass
    return broken_ids

def save_broken_ids(broken_ids):
    path = os.path.join(data_dir, 'broken_ids.txt')
    with open(path, 'w') as f:
        for l in broken_ids:
            f.write(f"{l}\n")

def load_thumbnails(vids):
    path = os.path.join(data_dir, 'thumbnails')
    Path(path).mkdir(parents=True, exist_ok=True)

    unique_vids = vids.groupby('video_id').first()
    num_downloaded = 0
    num_present = 0
    num_skipped = 0
    num_broken = 0
    broken_ids = load_broken_ids()
    session = requests.Session()
    for video_id, row in tqdm.tqdm(unique_vids.iterrows(), total=len(unique_vids)):
            if video_id in broken_ids:
                num_skipped += 1
                continue
            thumbnail_path = os.path.join(path, video_id + ".jpg")
            if os.path.exists(thumbnail_path):
                num_present += 1
                continue
            response = session.get(row.thumbnail_link)
            if response.status_code == 404:
                broken_ids.add(video_id)
                num_broken += 1
                continue
            with open(thumbnail_path, 'wb') as target:
                target.write(response.content)
            num_downloaded += 1
    if num_broken:
        save_broken_ids(broken_ids)
    print('Downloaded:', num_downloaded)
    print('Broken links:', num_broken)
    print('Already present:', num_present)
    print('Skipped:', num_skipped)

def each_thumbnail(limit=None):
    path = os.path.join(data_dir, 'thumbnails', '*')
    images = glob.iglob(path)
    margin = 11
    if limit:
        images = itertools.islice(images, limit)
    for filename in images:
        video_id = os.path.basename(filename)[:-4]
        im = cv2.imread(filename)
        yield video_id, im[margin:-margin, :]

def load_thumbnail_from_file(filename):
    margin = 11
    im = cv2.imread(filename)
    return im[margin:-margin, :]

def load_thumbnail(video_id):
    filename = os.path.join(data_dir, 'thumbnails', f"{video_id}.jpg")
    return load_thumbnail_from_file(filename)

def create_thumbs_df():
    path = os.path.join(data_dir, 'thumbnails', '*')
    images = glob.iglob(path)
    ids = [os.path.basename(filename)[:-4] for filename in images]
    return pd.DataFrame(data={ 'video_id': ids })
