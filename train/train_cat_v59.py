import numpy as np
import pandas as pd
import warnings
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.linear_model import LinearRegression, Ridge

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from pprint import pprint

from pathlib import Path
from tqdm import tqdm
from os import makedirs
from os.path import join
import argparse

import matplotlib
import matplotlib.pyplot as plt
try:
    import japanize_matplotlib
except:
    pass
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def rmse(pred, target):
    return np.sqrt(mean_squared_error(pred, target))


DATA_PATH = Path('./data')
PREFIX = 'v59'
Path('models').mkdir(exist_ok=True)
Path('predictions/train/').mkdir(exist_ok=True, parents=True)

SEED = 2019
FOLD = 10
ITER = 20000

ESSENTIAL_COLS = ['方角', '建物構造', 'adr0', 'adr1', 'adr2', 'adr3', 'adr4',
                  'age', 'area', 'floorplan', 'thisfloor', 'maxfloor', 'contr']
TRANSPORT_COLS = [f'{key}{i}' for i in range(
    3) for key in ['line', 'station', 'distance']]
COORD_COLS = [f'station{i}{coord}' for i in range(3) for coord in [
    '_lat', '_lon']]

USE_COLS = ESSENTIAL_COLS + [
    'floorfromtop',
    'station0', 'line0', 'distance0', 'station0_user', 'linecnts',
    #     'line_vec0', 'line_vec1', 'line_vec2',
    #     'trilat_lat', 'trilat_lon',
    'trilat2_lat', 'trilat2_lon',
    'lat', 'lng',
    #     'google_lat', 'google_lng',
    'landp',
    #     'landp_growth',
    'station0_user',
    'バス・トイレ別', '温水洗浄便座', '浴室乾燥機',
    'システムキッチン', 'エアコン付', '追焚機能', 'ウォークインクローゼット', 'ロフト付き', '防音室', 'インターネット対応', '冷蔵庫あり', 'ペアガラス', '床暖房',
    'エレベーター',
] + [
    '中央区_potential', '港区_potential',
    #     '中央区_potential', '渋谷区_potential',
    #     '大田区_potential', '港区_potential',
    #     '品川区_potential', '千代田区_potential',
    #     'park_bike', 'park_motor', 'park_car'
]

CAT_COLS = [
    '方角', '建物構造', 'adr0', 'adr1', 'adr2', 'adr3', 'adr4',
    'floorplan', 'contr'] + [
    'station0', 'line0',
    'バス・トイレ別', '温水洗浄便座', '浴室乾燥機',
    'システムキッチン', 'エアコン付', '追焚機能', 'ウォークインクローゼット', 'ロフト付き', '防音室', 'インターネット対応', '冷蔵庫あり', 'ペアガラス', '床暖房',
    'エレベーター',
    #     'park_bike', 'park_motor', 'park_car'
]

CAT_IDXS = [USE_COLS.index(col) for col in CAT_COLS]

CAT_PARAMS = {
    'iterations': ITER, 'one_hot_max_size': 3,
    'cat_features': CAT_IDXS,
    'use_best_model': True, 'eval_metric': 'MAE', 'thread_count': 4,
    'learning_rate': 0.1, 'random_state': SEED, 'depth': 6
}
CAT_FIT_PARAMS = {
    'early_stopping_rounds': 800, 'verbose': False, 'plot': False,
}


def main(load_model=False):
    # load data
    train = pd.read_csv(DATA_PATH / 'train_complete.csv', index_col=0)
    test = pd.read_csv(DATA_PATH / 'test_complete.csv', index_col=0)

    # get common_ids
    train_ids = set(train.buildingid.unique())
    test_ids = set(test.buildingid.unique())
    common_ids = train_ids & test_ids

    # make pseudo label
    print(f'[train] making pseudo labels')
    MATCH_COLS = ['area', 'thisfloor', 'buildingid']
    train_mini = train[['賃料'] +
                       MATCH_COLS].groupby(MATCH_COLS).agg(np.mean).reset_index()
    pseudo_label = test[MATCH_COLS]
    pseudo_label = pseudo_label.merge(train_mini, on=MATCH_COLS, how='left')
    pseudo_label.index = test.index

    null_idx = pseudo_label[pseudo_label['賃料'].isnull()].index
    MATCH_COLS = ['area', 'buildingid']

    multi_match = train.loc[train.buildingid.isin(common_ids),
                            ['賃料', 'thisfloor'] + MATCH_COLS].groupby(MATCH_COLS).agg(
        {'thisfloor': lambda x: len(x.unique()), '賃料': np.mean})
    single_match = multi_match.loc[multi_match['thisfloor'] == 1].reset_index()
    multi_match = multi_match.loc[multi_match['thisfloor'] > 1].reset_index()

    # Same bldg w/ one unique floor: fill mean value
    pseudo_label.loc[null_idx, '賃料'] = pseudo_label.drop('賃料', axis=1).loc[null_idx].merge(
        single_match, on=MATCH_COLS, how='left')['賃料'].values

    null_idx = pseudo_label['賃料'].isnull().values
    REG_COLS = ['thisfloor', '賃料']
    Xy_all = train.loc[train.buildingid.isin(multi_match.buildingid.values), REG_COLS + MATCH_COLS].merge(
        multi_match.drop(['賃料', 'thisfloor'], axis=1), on=MATCH_COLS, how='inner')
    X_test_all = test.reset_index().loc[test.buildingid.isin(multi_match.buildingid.values).values & null_idx,
                                        ['id'] + REG_COLS + MATCH_COLS].merge(
        multi_match.drop(['賃料', 'thisfloor'], axis=1), on=MATCH_COLS, how='inner').set_index('id')

    for (thisArea, thisId), group in tqdm(Xy_all.groupby(['area', 'buildingid'])):
        matched = X_test_all.loc[(X_test_all.area == thisArea) & (
            X_test_all.buildingid == thisId), 'thisfloor']
        x_test_index = matched.index
        x_test = matched.values.reshape(-1, 1)
        if len(x_test) == 0:
            continue
        x_train = group.values[:, 0].reshape(-1, 1)
        y = group.values[:, 1]
        reg = Ridge().fit(x_train, y)
        score = reg.score(x_train, y)
        if score > 0.9:
            y_test = reg.predict(x_test)
            pseudo_label.loc[x_test_index, '賃料'] = y_test

    add_test = pseudo_label.loc[~pseudo_label['賃料'].isnull(), '賃料']

    for col in CAT_COLS:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)

    # make dataset (train and pseudo label)
    add_train = test.loc[add_test.index].copy()
    add_train['賃料'] = add_test.values
    train = pd.concat([train, add_train])
    x_all = train[USE_COLS].values

    x_test = test[USE_COLS].values
    y_all = (train['賃料'] / train['area']).values
    price = train['賃料'].values
    area = train['area'].values
    test_area = test['area'].values

    # split dataset
    cv_split = GroupKFold(n_splits=FOLD).split(
        x_all, groups=train.buildingid.values)
    id_mask = (~train.buildingid.isin(common_ids)).astype(np.uint8).values

    # training models
    imps = np.zeros((x_all.shape[1], FOLD))
    scores = np.zeros(FOLD)
    oofs = np.zeros(len(y_all))
    preds = np.zeros(x_test.shape[0])
    for k, (train_idx, test_idx) in enumerate(cv_split):
        print(f'[train] starting fold {k}')
        x_train, x_valid = x_all[train_idx], x_all[test_idx]
        y_train, y_valid = y_all[train_idx], y_all[test_idx]
        price_train, area_train = price[train_idx], area[train_idx]
        price_valid, area_valid = price[test_idx], area[test_idx]
        mask_train, mask_valid = id_mask[train_idx], id_mask[test_idx]

        if load_model:
            model = CatBoostRegressor()
            model.load_model(join('./models', f'{PREFIX}_fold{k}.cbm'))
        else:
            train_data = Pool(
                data=x_train,
                label=y_train,
                #         weight = area_train,
                cat_features=CAT_IDXS
            )
            test_data = Pool(
                data=x_valid,
                label=y_valid,
                #         weight = area_valid,
                cat_features=CAT_IDXS
            )
            model = CatBoostRegressor(**CAT_PARAMS)
            model.fit(X=train_data, eval_set=test_data, **CAT_FIT_PARAMS)
            print(f'best iteration is {model.get_best_iteration()}/{ITER}')

            model.save_model(join('./models', f'{PREFIX}_fold{k}.cbm'))

        oofs[test_idx] = model.predict(x_valid)
        raw_score = rmse(oofs[test_idx], y_valid)
        score = rmse(oofs[test_idx] * area_valid, price_valid)
        print(f'score of fold {k} is {score:.3f} (raw {raw_score:.3f})')
        scores[k] = score
        preds += model.predict(x_test) / FOLD

        imps[:, k] = model.feature_importances_

    print(f'overall cv is {np.mean(scores):.3f} +- {np.std(scores):.3f}')

    if not load_model:
        plt.figure(figsize=(5, int(x_all.shape[1] / 3)))
        imps_mean = np.mean(imps, axis=1)
        imps_se = np.std(imps, axis=1) / np.sqrt(x_all.shape[0])
        order = np.argsort(imps_mean)
        plt.barh(np.array(USE_COLS)[order],
                 imps_mean[order], xerr=imps_se[order])
        plt.subplots_adjust(left=0.2)
        plt.savefig(join('./models', f'{PREFIX}_feature_importances.png'))

    s = rmse(oofs[train.index <= 31470] * area[train.index <= 31470],
             price[train.index <= 31470])
    print(f'all id train rent cv score:{s}')

    bachelor_ids = train.buildingid.value_counts(
    )[train.buildingid.value_counts() == 1].index
    s = rmse(oofs[~train.buildingid.isin(bachelor_ids).values] * area[~train.buildingid.isin(bachelor_ids).values],
             price[~train.buildingid.isin(bachelor_ids)])
    print(f'bachelor id train rent cv score:{s}')

    s = rmse(oofs[~train.buildingid.isin(common_ids).values] * area[~train.buildingid.isin(common_ids).values],
             price[~train.buildingid.isin(common_ids)])
    print(f'not common id train rent cv score:{s}')

    st2_preds = preds.copy()
    sub = pd.DataFrame(st2_preds * test.area, index=test.index)
    sub.to_csv(
        join('./predictions', f'{PREFIX}.csv'), index=True, header=False)

    train['prediction'] = oofs * train.area.values
    train['prediction'].to_csv(
        join('./predictions/train', f'{PREFIX}_train.csv'), index=True, header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", action='store_true',
                        help="run on inference mode")
    opt = parser.parse_args()
    print(f'[train] training {PREFIX} started')
    main(load_model=opt.inference)
