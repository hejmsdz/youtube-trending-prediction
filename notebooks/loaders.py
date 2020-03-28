import json
import os.path
import pandas as pd
import urllib.request
import tqdm
from pathlib import Path
from urllib.error import HTTPError

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

def load_thumbnails():
    path = os.path.join(data_dir, 'thumbnails')
    Path(path).mkdir(parents=True, exist_ok=True)

    vids = load_and_clean_up_videos()
    broken_links = []
    for _, row in tqdm.tqdm(vids.iterrows(), total=len(vids)):
        try:
            thumbnail_path = os.path.join(path, row.video_id + ".jpg")
            if not os.path.exists(thumbnail_path):
                urllib.request.urlretrieve(row.thumbnail_link, thumbnail_path)

        except HTTPError:
            broken_links.append(row.thumbnail_link)
            continue
    return broken_links
