import os.path
import pandas as pd

data_dir = '../youtube_data'

def load_csv(filename):
    path = os.path.join(data_dir, filename)
    return pd.read_csv(path, sep=';', encoding='macroman')

def load_gb_videos():
    return load_csv('GB_videos_5p.csv')

def load_us_videos():
    return load_csv('US_videos_5p.csv')
