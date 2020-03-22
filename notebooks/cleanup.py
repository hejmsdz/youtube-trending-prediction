import pandas as pd

def merge(gb_vids, us_vids):
    gb_vids['is_GB'] = True
    gb_vids['is_US'] = False

    us_vids['is_GB'] = False
    us_vids['is_US'] = True

    return pd.concat([gb_vids, us_vids])

def fill_missing_video_ids(vids):
    missing_ids = vids['video_id'] == '#NAZWA?'
    ids = vids[missing_ids]['thumbnail_link'].apply(lambda t: t[23:34])
    vids.loc[missing_ids, 'video_id'] = ids
    return vids

def remove_duplicates(vids):
    scope = ['video_id', 'trending_date']
    duplicates = vids.duplicated(vids.columns[:-2], keep=False)
    vids['duplicate'] = duplicates

    def markCountries(x):
        if x['duplicate']:
            x['is_GB'] = True
            x['is_US'] = True
        return x

    vids = vids.apply(markCountries, axis=1)
    vids.drop_duplicates(scope, inplace=True)
    vids.drop(columns=['duplicate'], inplace=True)
    return vids

def trim_column_names(vids):
    return vids.rename(columns={'description ': 'description'})
