import json
import os.path
import pandas as pd

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
    return vids

def load_all_videos():
    path = os.path.join(data_dir, 'all_videos.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    
    vids = load_and_clean_up_videos()
    vids.to_csv(path)
    return vids
