import json
import os.path
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
    num_images = 0
    broken_ids = load_broken_ids()
    session = requests.Session()
    for video_id, row in tqdm.tqdm(unique_vids.iterrows(), total=len(unique_vids)):
            if video_id in broken_ids:
                continue
            thumbnail_path = os.path.join(path, video_id + ".jpg")
            if os.path.exists(thumbnail_path):
                continue
            response = session.get(row.thumbnail_link)
            if response.status_code == 404:
                broken_ids.add(video_id)
                continue
            with open(thumbnail_path, 'wb') as target:
                target.write(response.content)
            num_images += 1
    save_broken_ids(broken_ids)
    return num_images
