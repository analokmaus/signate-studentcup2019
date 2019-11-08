from pathlib import Path
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

import mojimoji
import re
from preprocess import parse_address

import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path('data')
train = pd.read_csv(DATA_PATH / 'train_processed.csv', index_col=0)
test = pd.read_csv(DATA_PATH / 'test_processed.csv', index_col=0)

train.adr2 = train.adr2.fillna(0).astype(np.int).astype(np.str)
test.adr2 = test.adr2.fillna(0).astype(np.int).astype(np.str)
train.adr3 = train.adr3.fillna(0).astype(np.int).astype(np.str)
test.adr3 = test.adr3.fillna(0).astype(np.int).astype(np.str)
train.adr4 = train.adr4.fillna(0).astype(np.int).astype(np.str)
test.adr4 = test.adr4.fillna(0).astype(np.int).astype(np.str)

train['floorfromtop'] = train['maxfloor'] - train['thisfloor']
test['floorfromtop'] = test['maxfloor'] - test['thisfloor']

ESSENTIAL_COLS = ['賃料', '方角', '建物構造', 'adr0', 'adr1', 'adr2', 'adr3', 'adr4',
                  'age', 'area', 'floorplan', 'withstorage', 'thisfloor', 'maxfloor', 'contr']
TRANSPORT_COLS = [f'{key}{i}' for i in range(
    3) for key in ['line', 'station', 'distance']]
COORD_COLS = [f'station{i}{coord}' for i in range(3) for coord in [
    '_lat', '_lon']]
ADR_COLS = [f'adr{i}' for i in range(5)]

'''
Add building id
'''

print('[preprocess] adding building id')


def adr_fix(text):
    if text != text:
        return np.nan
    text = text.replace('ヶ', 'ケ')
    return text


train['adr1'] = train['adr1'].map(adr_fix)
train['adr1'] = train['adr1'].replace({'三栄町': '四谷三栄町'})
test['adr1'] = test['adr1'].map(adr_fix)
test['adr1'] = test['adr1'].replace({'三栄町': '四谷三栄町'})

ID_COL = ['adr1', 'adr2', '建物構造', 'age', 'maxfloor']
train['buildingid'] = ''
test['buildingid'] = ''
for col in ID_COL:
    if col not in ['age', 'maxfloor']:
        train['buildingid'] += train[col].fillna(0).astype(str).str[:5]
        test['buildingid'] += test[col].fillna(0).astype(str).str[:5]
    elif col in ['age', 'maxfloor']:
        train['buildingid'] += col[0] + \
            (train[col].fillna(-1) * 100).astype(int).astype(str)
        test['buildingid'] += col[0] + \
            (test[col].fillna(-1) * 100).astype(int).astype(str)


'''
merge id
面積と住所で同じ建物を検出する
'''


def step_search(ages, step=0.2):
    _ages = np.argsort(ages)
    border = ages[_ages[0]]  # min
    group = np.zeros(len(ages), dtype=np.uint8)
    now = 0
    for idx in _ages:
        if ages[idx] <= border + step:
            group[idx] = now
            border = ages[idx]
        else:
            now += 1
            group[idx] = now
            border = ages[idx]
    return group


def isSameBldg(adr_group):
    criterion = np.array([
        # area XX.YY(Y!=0), age_diff is integer
        adr_group.area.values[0] - int(adr_group.area.values[0]) > 0 and \
        (adr_group.age.max() - adr_group.age.min()
         ).is_integer() and adr_group.age.max() - adr_group.age.min() <= 7,
        # age diff <= 1, same max floor
        adr_group.age.max() - adr_group.age.min() <= 1.0 and len(adr_group.maxfloor.unique()) == 1,
        # age diff <= 2, same station and distance
        adr_group.age.max() - adr_group.age.min() <= 2.0 and \
        len(adr_group.station0.unique()) == 1 and len(
            adr_group.distance0.unique()) == 1,
        # age diff == 1 or <= 0.5 (typo)
        adr_group.age.max() - adr_group.age.min() == 1.0 or adr_group.age.max() - \
        adr_group.age.min() <= 0.5,
    ])
    if np.sum(criterion) >= 1:
        return True
    else:
        return False


failed_ids = []
for i, (area_val, area_group) in enumerate(tqdm(pd.concat([train, test]).groupby('area'))):
    if len(area_group) == 1:
        continue

    for adr_val, adr_group in area_group.groupby(['adr1', 'adr2']):
        if len(adr_group.buildingid.unique()) == 1:
            continue
        if isSameBldg(adr_group):
            replace_dict = {
                k: adr_group.buildingid.values[0] for k in adr_group.buildingid.values}
            train.buildingid = train.buildingid.replace(replace_dict)
            test.buildingid = test.buildingid.replace(replace_dict)
            continue

        if len(adr_group) >= 3:
            group = step_search(adr_group.age.values, step=1.0)
            for g in range(max(group) + 1):
                cluster = adr_group.iloc[group == g]
                if len(cluster) == 1 or len(cluster.buildingid.unique()) == 1:
                    continue
                if isSameBldg(cluster):
                    replace_dict = {
                        k: cluster.buildingid.values[0] for k in cluster.buildingid.values}
                    train.buildingid = train.buildingid.replace(replace_dict)
                    test.buildingid = test.buildingid.replace(replace_dict)

        failed_ids.append(f'{area_val}{adr_val}')

for i, (floor_val, floor_group) in enumerate(tqdm(pd.concat([train, test]).groupby('maxfloor'))):
    if floor_val < 5 or len(floor_group) == 1:
        continue

    for adr_val, adr_group in floor_group.groupby(['adr1', 'adr2']):
        if len(adr_group.buildingid.unique()) == 1:
            continue

        if len(adr_group) >= 2:
            group = step_search(adr_group.age.values, step=0.3)
            for g in range(max(group) + 1):
                cluster = adr_group.iloc[group == g]
                if len(cluster) == 1 or len(cluster.buildingid.unique()) == 1:
                    continue

                replace_dict = {
                    k: cluster.buildingid.values[0] for k in cluster.buildingid.values}
                train.buildingid = train.buildingid.replace(replace_dict)
                test.buildingid = test.buildingid.replace(replace_dict)

for i, (thisAdr, adr_group) in enumerate(tqdm(pd.concat([train, test]).groupby(ADR_COLS))):
    if '0' in thisAdr:
        continue

    if len(adr_group.buildingid.unique()) != 1:
        replace_dict = {
            k: adr_group.buildingid.values[0] for k in adr_group.buildingid.values}
        train.buildingid = train.buildingid.replace(replace_dict)
        test.buildingid = test.buildingid.replace(replace_dict)

train_ids = set(train.buildingid.unique())
test_ids = set(test.buildingid.unique())
print(
    f'train ids: {len(train_ids)} / test ids: {len(test_ids)}\ncommon ids: {len(train_ids & test_ids)}')
common_ids = train_ids & test_ids

'''
building info fix
'''

FIX_COLS = ['adr3', 'adr4', '建物構造', 'age',
            'maxfloor', 'contr'] + TRANSPORT_COLS
for i, (bldgid, bldg_group) in enumerate(tqdm(pd.concat([train, test]).groupby('buildingid'))):

    if len(bldg_group) == 1:
        continue
    train_idx = bldg_group.index[bldg_group.index <= 31470]
    test_idx = bldg_group.index[bldg_group.index > 31470]
    for col in FIX_COLS:
        vals = bldg_group[col].dropna().values
        if len(np.unique(vals)) >= 2:
            if 'adr' in col:
                valid_vals = vals[vals != '0']
                if len(valid_vals) >= 1:
                    true_val = valid_vals[0]
                else:
                    true_val = '0'
            else:
                true_val = stats.mode(vals)[0]

            train.loc[train_idx, col] = true_val
            test.loc[test_idx, col] = true_val

'''
floorplan typo fix
'''

print('[preprocess] fixing floorplan')
for i, (vals, group) in enumerate(tqdm(pd.concat([train, test]).groupby(['buildingid', 'area']))):

    if len(group) == 1:
        continue

    train_idx = group.index[group.index <= 31470]
    test_idx = group.index[group.index > 31470]

    if len(group.floorplan.unique()) > 1 and len(group) >= 3:
        fplan_vals = group.floorplan.dropna().values
        true_val = stats.mode(fplan_vals)[0]

        train.loc[train_idx, 'floorplan'] = true_val
        test.loc[test_idx, 'floorplan'] = true_val

bachelor_ids = train.buildingid.value_counts(
)[train.buildingid.value_counts() == 1].index
bachelor_train = train.loc[(~train.buildingid.isin(
    common_ids) & train.buildingid.isin(bachelor_ids))]
bachelor_test = test.loc[~test.buildingid.isin(common_ids)]

'''
Add coordinates info
'''

print('[preprocess] adding coordinate features')
train = train.drop(
    [x for x in train.columns if 'lat' in x or 'lng' in x], axis=1)
test = test.drop([x for x in test.columns if 'lat' in x or 'lng' in x], axis=1)

gps = pd.read_csv(DATA_PATH / '13_2018.csv', encoding='cp932')


def parse_gps_address(text):
    kan_to_ara = str.maketrans('一二三四五六七八九', '123456789')
    _chome = re.search(r'[一二三四五六七八九]+丁目', text)
    if _chome:
        chome = _chome.group()[:-2]
        chome = chome.translate(kan_to_ara)
        return text.replace(_chome.group(), ''), chome
    else:
        return text, np.nan


gps['adr0'] = gps['都道府県名'] + gps['市区町村名']
new_gps = gps[['adr0', '緯度', '経度']]
adr12 = gps['大字町丁目名'].apply(lambda x: pd.Series(parse_gps_address(x)))
adr12.columns = ['adr1', 'adr2']
new_gps = new_gps.merge(adr12, left_index=True, right_index=True).rename(
    {'緯度': 'lat', '経度': 'lng'}, axis=1)

train = train.reset_index().merge(
    new_gps, on=['adr0', 'adr1', 'adr2'], how='left').set_index('id')
# 町名でマージ
new_gps2 = new_gps.groupby(['adr0', 'adr1']).agg(np.mean).reset_index()
train.loc[train.lat.isnull()] = train.loc[train.lat.isnull()].drop(['lat', 'lng'], axis=1).merge(
    new_gps2, on=['adr0', 'adr1'], how='left').values
# それでもダメなら最寄駅で穴埋め
sta_gps = train.loc[train.station0.isin(train.loc[train.lat.isnull()].station0.values)].groupby('station0').agg(
    {'lat': np.nanmean, 'lng': np.nanmean}).reset_index()
train.loc[train.lat.isnull()] = train.loc[train.lat.isnull()].drop(['lat', 'lng'], axis=1).merge(
    sta_gps, on='station0', how='left').values
test = test.reset_index().merge(
    new_gps, on=['adr0', 'adr1', 'adr2'], how='left').set_index('id')
# 町名でマージ
new_gps2 = new_gps.groupby(['adr0', 'adr1']).agg(np.mean).reset_index()
test.loc[test.lat.isnull()] = test.loc[test.lat.isnull()].drop(['lat', 'lng'], axis=1).merge(
    new_gps2, on=['adr0', 'adr1'], how='left').values
train = train.drop([x for x in train.columns if 'google' in x], axis=1)
test = test.drop([x for x in test.columns if 'google' in x], axis=1)

google_gps_train = pd.read_csv(
    DATA_PATH / 'google_map_train.csv', index_col=0)[['id', 'lat', 'lng']].set_index('id')
google_gps_train.index += 1
google_gps_train.columns = ['google_lat', 'google_lng']
google_gps_train = google_gps_train.merge(
    train[ADR_COLS], left_index=True, right_index=True, how='inner')
google_gps_train = google_gps_train.drop_duplicates(ADR_COLS)

google_gps_test = pd.read_csv(
    DATA_PATH / 'google_map_test.csv', index_col=0)[['id', 'lat', 'lng']].set_index('id')
google_gps_test.index += 31471
google_gps_test.columns = ['google_lat', 'google_lng']
google_gps_test = google_gps_test.merge(
    test[ADR_COLS], left_index=True, right_index=True, how='inner')
google_gps_test = google_gps_test.drop_duplicates(ADR_COLS)

train = train.reset_index().merge(
    google_gps_train, on=ADR_COLS, how='left').set_index('id')
train.loc[train.google_lat.isnull(), ['google_lat', 'google_lng']
          ] = train.loc[train.google_lat.isnull(), ['lat', 'lng']].values
test = test.reset_index().merge(
    google_gps_test, on=ADR_COLS, how='left').set_index('id')
test.loc[test.google_lat.isnull(), ['google_lat', 'google_lng']
         ] = test.loc[test.google_lat.isnull(), ['lat', 'lng']].values

'''
Add station gps
'''

print('[preprocess] adding station features')
sta_info_tokyo = pd.read_csv(DATA_PATH / 'csv_roseneki_13.csv')
sta_info_saitama = pd.read_csv(DATA_PATH / 'csv_roseneki_11.csv')
sta_info_chiba = pd.read_csv(DATA_PATH / 'csv_roseneki_12.csv')
sta_info_kanagawa = pd.read_csv(DATA_PATH / 'csv_roseneki_12.csv')
sta_info = pd.concat([sta_info_tokyo, sta_info_saitama,
                      sta_info_chiba, sta_info_kanagawa])
sta_info.station_name = sta_info.station_name + '駅'
sta_info = sta_info[['pref_code', 'station_name', 'line_name',
                     'station_lat', 'station_lon']].reset_index(drop=True)

sta_unique = sta_info.groupby('station_name').agg(
    {'station_lat': lambda x: len(x.unique())})['station_lat']
sta_unique = sta_unique[sta_unique > 1].index
for sta in sta_unique:
    prefs = sta_info.loc[sta_info.station_name == sta, 'pref_code']
    if len(prefs.unique()) > 1 and 13 in prefs.values:
        kill_idx = prefs.index[prefs != 13]
        sta_info = sta_info.drop(kill_idx)
sta_info = sta_info.drop('pref_code', axis=1)

sta_info.loc[(sta_info.line_name == "つくばエクスプレス") & (
    sta_info.station_name == "浅草駅"), 'station_name'] = '浅草(ＴＸ)駅'
sta_info.loc[(sta_info.line_name == "日暮里・舎人ライナー") & (
    sta_info.station_name == "熊野前駅"), 'station_name'] = '熊野前(舎人ライナー)駅'
sta_info.loc[(sta_info.line_name == "都営地下鉄大江戸線") & (
    sta_info.station_name == "両国駅"), 'station_name'] = '両国(都営線)駅'
sta_info.loc[(sta_info.line_name == "西武豊島線") & (
    sta_info.station_name == "豊島園駅"), 'station_name'] = '豊島園(西武線)駅'
sta_info.loc[(sta_info.line_name == "都営地下鉄大江戸線") & (
    sta_info.station_name == "豊島園駅"), 'station_name'] = '豊島園(都営線)駅'
sta_info.loc[(sta_info.line_name == "都電荒川線") & (
    sta_info.station_name == "早稲田駅"), 'station_name'] = '早稲田(都電荒川線)駅'
sta_info.loc[(sta_info.line_name == "京成本線") & (
    sta_info.station_name == "町屋駅"), 'station_name'] = '町屋(京成線)駅'
sta_info.loc[(sta_info.line_name == "日暮里・舎人ライナー") & (
    sta_info.station_name == "西日暮里駅"), 'station_name'] = '西日暮里(舎人ライナー)駅'
sta_info.loc[(sta_info.line_name == "日暮里・舎人ライナー") & (
    sta_info.station_name == "日暮里駅"), 'station_name'] = '日暮里(舎人ライナー)駅'
sta_info.loc[(sta_info.line_name == "ＪＲ総武本線") & (
    sta_info.station_name == "本八幡駅"), 'station_name'] = '本八幡(総武線)駅'

sta_linecnt = sta_info[['station_name', 'line_name']].groupby(
    'station_name').agg('count').reset_index()
sta_gps = sta_info[['station_name', 'station_lat', 'station_lon']].groupby(
    'station_name').agg(np.mean).reset_index()

for col in [f'station{i}' for i in range(3)]:
    sta_gps.columns = [col, f'{col}_lat', f'{col}_lon']
    train[col] = train[col].replace({'市ヶ谷駅': '市ケ谷駅'})
    test[col] = test[col].replace({'市ヶ谷駅': '市ケ谷駅'})
    train = train.reset_index().merge(sta_gps, on=col, how='left').set_index('id')
    test = test.reset_index().merge(sta_gps, on=col, how='left').set_index('id')

train = train.drop([x for x in train.columns if 'linecnt' in x], axis=1)
test = test.drop([x for x in test.columns if 'linecnt' in x], axis=1)

for col in [f'station{i}' for i in range(3)]:
    sta_linecnt.columns = [col, f'{col}_linecnt']
    train[col] = train[col].replace({'市ヶ谷駅': '市ケ谷駅'})
    test[col] = test[col].replace({'市ヶ谷駅': '市ケ谷駅'})
    train = train.reset_index().merge(sta_linecnt, on=col, how='left').set_index('id')
    test = test.reset_index().merge(sta_linecnt, on=col, how='left').set_index('id')
    train[f'{col}_linecnt'] = train[f'{col}_linecnt'].fillna(0).astype(np.int)
    test[f'{col}_linecnt'] = test[f'{col}_linecnt'].fillna(0).astype(np.int)

train['linecnts'] = train['station0_linecnt'] + \
    train['station1_linecnt'] + train['station2_linecnt']
test['linecnts'] = test['station0_linecnt'] + \
    test['station1_linecnt'] + test['station2_linecnt']

'''
Trilateration
'''

print('[preprocess] doing trilateration')
train = train.drop([x for x in train.columns if 'trilat' in x], axis=1)
test = test.drop([x for x in test.columns if 'trilat' in x], axis=1)


def trilat_simple(xs, ys, rs):
    ys = ys[xs == xs]
    rs = rs[xs == xs]
    xs = xs[xs == xs]
    r_sum = np.sum(rs)
    x = 0
    y = 0
    for i in range(len(xs)):
        x += xs[i] * rs[-i - 1] / r_sum
        y += ys[i] * rs[-i - 1] / r_sum

    return x, y


DIST_CONST = 80 / np.sqrt(2) / 1000


def trilat_precise(LatA, LonA, DistA, LatB, LonB, DistB, LatC, LonC, DistC):
    earthR = 6371  # km

    DistA = DistA * DIST_CONST  # km
    DistB = DistB * DIST_CONST  # km
    DistC = DistC * DIST_CONST  # km

    # authalic sphere
    xA = earthR * (math.cos(math.radians(LatA)) * math.cos(math.radians(LonA)))
    yA = earthR * (math.cos(math.radians(LatA)) * math.sin(math.radians(LonA)))
    zA = earthR * (math.sin(math.radians(LatA)))

    xB = earthR * (math.cos(math.radians(LatB)) * math.cos(math.radians(LonB)))
    yB = earthR * (math.cos(math.radians(LatB)) * math.sin(math.radians(LonB)))
    zB = earthR * (math.sin(math.radians(LatB)))

    xC = earthR * (math.cos(math.radians(LatC)) * math.cos(math.radians(LonC)))
    yC = earthR * (math.cos(math.radians(LatC)) * math.sin(math.radians(LonC)))
    zC = earthR * (math.sin(math.radians(LatC)))

    P1 = np.array([xA, yA, zA])
    P2 = np.array([xB, yB, zB])
    P3 = np.array([xC, yC, zC])

    # transform to get circle 1 at origin
    # transform to get circle 2 on x axis
    ex = (P2 - P1) / (np.linalg.norm(P2 - P1))
    i = np.dot(ex, P3 - P1)
    ey = (P3 - P1 - i * ex) / (np.linalg.norm(P3 - P1 - i * ex))
    ez = np.cross(ex, ey)
    d = np.linalg.norm(P2 - P1)
    j = np.dot(ey, P3 - P1)

    # plug and chug using above values
    x = (DistA**2 - DistB**2 + d**2) / (2 * d)
    y = ((DistA**2 - DistC**2 + i**2 + j**2) / (2 * j)) - ((i / j) * x)

    # only one case shown here
    if DistA**2 - x**2 - y**2 > 0:
        z = np.sqrt(DistA**2 - x**2 - y**2)
    else:
        z = 0

    # triPt is an array with ECEF x,y,z of trilateration point
    diff = x * ex + y * ey + z * ez
    if np.linalg.norm(diff) > DistA:
        diff = DistA * diff / np.linalg.norm(diff)
#         return np.nan, np.nan
    triPt = P1 + diff

    # convert back to lat/long from ECEF
    # convert to degrees
    try:
        lat = math.degrees(math.asin(triPt[2] / earthR))
        lon = math.degrees(math.atan2(triPt[1], triPt[0]))
    except:
        print(triPt)
        return np.nan, np.nan

    return lat, lon


TRILAT_COLS = ['station0_lat', 'station0_lon', 'distance0',
               'station1_lat', 'station1_lon', 'distance1',
               'station2_lat', 'station2_lon', 'distance2']

new_df = []
for coord in tqdm(train[TRILAT_COLS].values):
    tmp = []
    tmp.extend(trilat_precise(*coord))
    tmp.extend(trilat_simple(coord[[0, 3, 6]],
                             coord[[1, 4, 7]], coord[[2, 5, 8]]))
    new_df.append(tmp)
new_df = pd.DataFrame(new_df, index=train.index,
                      columns=['trilat_lat', 'trilat_lon', 'trilat2_lat', 'trilat2_lon'])
train = train.merge(new_df, left_index=True, right_index=True)

new_df = []
for coord in tqdm(test[TRILAT_COLS].values):
    tmp = []
    tmp.extend(trilat_precise(*coord))
    tmp.extend(trilat_simple(coord[[0, 3, 6]],
                             coord[[1, 4, 7]], coord[[2, 5, 8]]))
    new_df.append(tmp)
new_df = pd.DataFrame(new_df, index=test.index,
                      columns=['trilat_lat', 'trilat_lon', 'trilat2_lat', 'trilat2_lon'])
test = test.merge(new_df, left_index=True, right_index=True)

null_idx = train.trilat_lat.isnull()
train.loc[null_idx, 'trilat_lat'] = train.loc[null_idx, 'lat']
train.loc[null_idx, 'trilat_lon'] = train.loc[null_idx, 'lng']
null_idx = test.trilat_lat.isnull()
test.loc[null_idx, 'trilat_lat'] = test.loc[null_idx, 'lat']
test.loc[null_idx, 'trilat_lon'] = test.loc[null_idx, 'lng']

'''
区ポテンシャル
'''

print('[preprocess] adding ward potential features')
for ward in tqdm(train.adr0.unique()):
    ward_pos = np.mean(train.loc[train.adr0 == ward, [
                       'lng', 'lat']].values, axis=0)
    train[f'{ward[3:]}_potential'] = train[['lng', 'lat']].apply(
        lambda x: 1 / np.linalg.norm(x.values - ward_pos), axis=1)
    test[f'{ward[3:]}_potential'] = test[['lng', 'lat']].apply(
        lambda x: 1 / np.linalg.norm(x.values - ward_pos), axis=1)

'''
Clean floorplan
'''

print('[preprocess] doing additional cleansing')
floorplan_dict = {
    # Name fix
    '11R': '1R', '1LK': '1LDK', '2R': '1R',
    # Merge
    '4DK': '>4DK/K', '4K': '>4DK/K',
    '5DK': '>4DK/K', '5K': '>4DK/K',
    '6DK': '>4DK/K', '6K': '>4DK/K',
    '5LDK': '>5LDK', '6LDK': '>5LDK', '8LDK': '>5LDK',
}

train['floorplan'] = train['floorplan'].replace(floorplan_dict)
test['floorplan'] = test['floorplan'].replace(floorplan_dict)

'''
Floorfix
'''

train.loc[train.thisfloor > train.maxfloor,
          'thisfloor'] = train.loc[train.thisfloor > train.maxfloor, 'maxfloor'].values
test.loc[test.thisfloor > test.maxfloor,
         'thisfloor'] = test.loc[test.thisfloor > test.maxfloor, 'maxfloor'].values

'''
Add land price
'''

print('[preprocess] adding land price features')
landp30 = pd.read_csv(DATA_PATH / 'L02-30P-2K_13.csv', encoding='cp932')[['住居表示', 'Ｈ３０価格']].rename(
    columns={'Ｈ３０価格': '価格'})
landp28 = pd.read_csv(DATA_PATH / 'L02-28P-2K_13.csv', encoding='cp932')[['住居表示', 'Ｈ２８価格']].rename(
    columns={'Ｈ２８価格': '価格'})


def parse_kansuji(text):
    kan_to_ara = str.maketrans('一二三四五六七八九', '123456789')
    _chome = re.search(r'[一二三四五六七八九]+丁目', text)
    if _chome:
        chome = _chome.group()
        chome = chome.translate(kan_to_ara)
        return text.replace(_chome.group(), chome)
    return text


def clean_landprice(landp):
    landp = landp.rename(
        columns={'価格': 'landp', '当年価格（円）': 'landp', '前年価格（円）': 'landp_prev'})
    landp = landp.loc[landp['住居表示'].str.contains('区')]
#     landp_adr = landp['住居表示'].apply(lambda x: pd.Series(parse_address(parse_kansuji(x))))
    landp_adr = landp['住居表示'].apply(lambda x: pd.Series(parse_address(x)))
    landp_adr.columns = [f'adr{i}' for i in range(5)]
    landp = landp.merge(landp_adr, left_index=True,
                        right_index=True).drop('住居表示', axis=1)
    return landp


landp30 = clean_landprice(landp30)
landp28 = clean_landprice(landp28)
landp = landp30.merge(
    landp28, on=[f'adr{i}' for i in range(5)], suffixes=('_30', '_28'))

landp['diff'] = landp['landp_30'] / landp['landp_28']

train['landp'] = 0
test['landp'] = 0
train['landp_growth'] = 1
test['landp_growth'] = 1

# adr0
tmp_landp = landp.drop(['adr4', 'adr3', 'adr2', 'adr1'], axis=1).groupby(
    'adr0').agg(np.mean).reset_index()
tmp = train.drop('landp', axis=1).reset_index().merge(
    tmp_landp, on=['adr0']).set_index('id')
train.loc[tmp.index, 'landp'] = tmp['landp_30'].values
train.loc[tmp.index, 'landp_growth'] = tmp['diff'].values

# adr0-1
tmp_landp = landp.drop(['adr4', 'adr3', 'adr2'], axis=1).groupby(
    [f'adr{i}' for i in range(2)]).agg(np.mean).reset_index()
tmp = train.drop('landp', axis=1).reset_index().merge(
    tmp_landp, on=[f'adr{i}' for i in range(2)]).set_index('id')
train.loc[tmp.index, 'landp'] = tmp['landp_30'].values
train.loc[tmp.index, 'landp_growth'] = tmp['diff'].values

# adr0-2
tmp_landp = landp.drop(['adr4', 'adr3'], axis=1).groupby(
    [f'adr{i}' for i in range(3)]).agg(np.mean).reset_index()
tmp = train.drop('landp', axis=1).reset_index().merge(
    tmp_landp, on=[f'adr{i}' for i in range(3)]).set_index('id')
train.loc[tmp.index, 'landp'] = tmp['landp_30'].values
train.loc[tmp.index, 'landp_growth'] = tmp['diff'].values

# adr0
tmp_landp = landp.drop(['adr4', 'adr3', 'adr2', 'adr1'], axis=1).groupby(
    'adr0').agg(np.mean).reset_index()
tmp = test.drop('landp', axis=1).reset_index().merge(
    tmp_landp, on=['adr0']).set_index('id')
test.loc[tmp.index, 'landp'] = tmp['landp_30'].values
test.loc[tmp.index, 'landp_growth'] = tmp['diff'].values

# adr0-1
tmp_landp = landp.drop(['adr4', 'adr3', 'adr2'], axis=1).groupby(
    [f'adr{i}' for i in range(2)]).agg(np.mean).reset_index()
tmp = test.drop('landp', axis=1).reset_index().merge(
    tmp_landp, on=[f'adr{i}' for i in range(2)]).set_index('id')
test.loc[tmp.index, 'landp'] = tmp['landp_30'].values
test.loc[tmp.index, 'landp_growth'] = tmp['diff'].values

# adr0-2
tmp_landp = landp.drop(['adr4', 'adr3'], axis=1).groupby(
    [f'adr{i}' for i in range(3)]).agg(np.mean).reset_index()
tmp = test.drop('landp', axis=1).reset_index().merge(
    tmp_landp, on=[f'adr{i}' for i in range(3)]).set_index('id')
test.loc[tmp.index, 'landp'] = tmp['landp_30'].values
test.loc[tmp.index, 'landp_growth'] = tmp['diff'].values

'''
Add station user info
'''

print('[preprocess] adding other features')
train = train.drop([x for x in train.columns if 'user' in x], axis=1)
test = test.drop([x for x in test.columns if 'user' in x], axis=1)

sta_user = pd.read_csv(DATA_PATH / 'station_users.csv', encoding='cp932')
sta_user['駅名'] = sta_user['駅名'] + '駅'
sta_user = sta_user.rename(columns={'乗降客数': 'sta_user', '駅名': 'station'})[
    ['sta_user', 'station']]

sta_user = sta_user.groupby('station').agg(np.sum).reset_index()
sta_user.columns = ['station0', 'station0_user']

train = train.reset_index().merge(
    sta_user, on='station0', how='left').set_index('id')
test = test.reset_index().merge(sta_user, on='station0', how='left').set_index('id')

train['station0_user'] = train['station0_user'].fillna(0).astype(np.int)
test['station0_user'] = test['station0_user'].fillna(0).astype(np.int)

'''
Add population
'''

train = train.drop([x for x in train.columns if 'population' in x], axis=1)
test = test.drop([x for x in test.columns if 'population' in x], axis=1)

ppl = pd.read_csv(DATA_PATH / 'tokyo_population.csv',
                  encoding='cp932', usecols=[2, 3, 7], skiprows=1)
ppl = ppl.drop(0)
ppl.columns = ['_adr0', '_adr1', 'population']
ppl._adr0 = '東京都' + ppl._adr0
ppl['adr'] = ppl._adr0 + ppl._adr1
ppl = ppl[['adr', 'population']]
ppl = ppl.loc[(~ppl.adr.isnull()) & ppl.adr.str.contains('区')]
ppl = ppl.merge(ppl.adr.apply(lambda x: pd.Series(
    parse_address(x))), left_index=True, right_index=True)
ppl = ppl.drop(['adr', 3, 4], axis=1)
ppl.columns = ['population', 'adr0', 'adr1', 'adr2']
ppl = ppl.loc[ppl.adr2 != '']
ppl = ppl.loc[ppl.population.str.isnumeric()]
ppl.population = ppl.population.astype(np.float16)
ppl = ppl.groupby([f'adr{i}' for i in range(3)]).mean().reset_index()

train = train.reset_index().merge(
    ppl, on=[f'adr{i}' for i in range(3)], how='left').set_index('id')
test = test.reset_index().merge(
    ppl, on=[f'adr{i}' for i in range(3)], how='left').set_index('id')

tmp_ppl = ppl.drop(['adr2'], axis=1).groupby(
    [f'adr{i}' for i in range(2)]).agg(np.mean).reset_index()
tmp = train.drop('population', axis=1).reset_index().merge(
    tmp_ppl, on=[f'adr{i}' for i in range(2)], how='left').set_index('id')

target_idx = train.index.isin(tmp.index)
train.loc[target_idx, 'population'] = tmp.loc[target_idx, 'population'].values

tmp_ppl = ppl.drop(['adr2'], axis=1).groupby(
    [f'adr{i}' for i in range(2)]).agg(np.mean).reset_index()
tmp = test.drop('population', axis=1).reset_index().merge(
    tmp_ppl, on=[f'adr{i}' for i in range(2)], how='left').set_index('id')
target_idx = test.index.isin(tmp.index)
test.loc[target_idx, 'population'] = tmp.loc[target_idx, 'population'].values

train.to_csv(DATA_PATH / 'train_complete.csv')
test.to_csv(DATA_PATH / 'test_complete.csv')

print('[preprocess] preprocess completed')
