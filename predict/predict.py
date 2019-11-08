import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pprint import pprint

from pathlib import Path
from tqdm import tqdm


def rmse(pred, target):
    return np.sqrt(mean_squared_error(pred, target))


warnings.filterwarnings('ignore')


'''
Load data
'''

DATA_PATH = Path('data')
train = pd.read_csv(DATA_PATH / 'train_complete.csv', index_col=0)
test = pd.read_csv(DATA_PATH / 'test_complete.csv', index_col=0)

train_ids = set(train.buildingid.unique())
test_ids = set(test.buildingid.unique())
common_ids = train_ids & test_ids

oof_files = list(Path('predictions/train/').glob('*_train.csv'))

include = [
    'v56',
    'v58',
    'v59'
]

for oof_f in oof_files:
    name = oof_f.stem[:-6]
    if name not in include:
        continue
    tmp = pd.read_csv(oof_f, index_col=0)['prediction']
    if tmp.shape[0] != 31470:
        tmp = tmp.loc[train.index]
    train[f'pred.{name}'] = tmp.values

    sub = pd.read_csv(f'predictions/{name}.csv', index_col=0, header=None)
    test[f'pred.{name}'] = sub[1].values
    print(f'[predict] {name} loaded')

'''
Stratified stacking
'''

print('[predict] stratified stacking')
ESSENTIAL_COLS = ['方角', '建物構造', 'adr0', 'adr1', 'adr2', 'adr3', 'adr4',
                  'age', 'area', 'floorplan', 'thisfloor', 'maxfloor', 'contr']
TRANSPORT_COLS = [f'{key}{i}' for i in range(
    3) for key in ['line', 'station', 'distance']]
COORD_COLS = [f'station{i}{coord}' for i in range(3) for coord in [
    '_lat', '_lon']]
PRED_COLS = [col for col in train.columns if 'pred.' in col]

train['avg_pred'] = train[PRED_COLS].mean(axis=1)
test['avg_pred'] = test[PRED_COLS].mean(axis=1)

USE_COLS = PRED_COLS

y_oof = train['avg_pred'].values.copy()
y_test = test['avg_pred'].values.copy()

for i, (thisAdr, group) in enumerate(pd.concat([train, test]).groupby(['adr0'])):
    train_idx = group.index[group.index <= 31470]
    test_idx = group.index[group.index > 31470]
    if len(train_idx) == 0 or len(test_idx) == 0:
        continue

    X = group.loc[train_idx, USE_COLS].fillna(0).values
    X_test = group.loc[test_idx, USE_COLS].fillna(0).values
    y = group.loc[train_idx, '賃料'].values
    y_baseline = group.loc[train_idx, 'avg_pred'].values

    reg = Ridge(alpha=10, fit_intercept=0, random_state=0)
    reg.fit(X, y)
    if reg.score(X, y) > 0.5:
        y_pred = reg.predict(X)
        y_oof[train_idx - 1] = reg.predict(X)
        y_test[test_idx - 31471] = reg.predict(X_test)
        # print(int(rmse(y, y_baseline)), '->', int(rmse(y, y_pred)))
        # print(f'y = {reg.intercept_} + \n{reg.coef_.astype(np.float16)}\n{USE_COLS}\n')

cv = rmse(train['賃料'], y_oof)
print('cv', cv)

train['avg_pred'] = train[PRED_COLS].mean(axis=1)
test['avg_pred'] = test[PRED_COLS].mean(axis=1)
train['stack_pred'] = y_oof
test['stack_pred'] = y_test


'''
Adaptive stacking and export
'''

print('[predict] adaptive stacking')
test_fill = pd.read_csv(DATA_PATH / 'test_fill.csv', index_col=0, header=None)
submission = pd.DataFrame(y_test, index=test.index, columns=['pred']).copy()
submission.loc[test_fill.index, 'pred'] = test_fill[1].values
submission.head()


def make_match_df(train, test, match_cols):
    train2 = train[MATCH_COLS]
    test2 = test[MATCH_COLS]
    sc = StandardScaler()
    sc.fit(train2.values)
    train2.loc[:, MATCH_COLS] = sc.transform(train2.values)
    test2.loc[:, MATCH_COLS] = sc.transform(test2.values)
    return train2, test2


def adaptive_stacking(idx, num=10):
    global REG_COLS
    scores = np.abs((test2.loc[idx] - train2)).sum(axis=1)
    tmp = train.loc[scores.sort_values().head(num).index].fillna(0)
    tmp['scores'] = scores.sort_values().head(num)

    sc = StandardScaler()
    X = tmp[REG_COLS].values
    X_test = test.loc[idx, REG_COLS].fillna(0).values.reshape(1, -1)
    sc.fit(X)
    X = sc.transform(X)
    X_test = sc.transform(X_test)
    y = tmp['賃料'].values

    model = LinearRegression()
    model.fit(X, y)

    if model.score(X, y) > 0.6:
        return model.predict(X_test)[0]
    else:
        return submission.loc[idx, 'pred']


MATCH_COLS = ['trilat2_lat', 'trilat2_lon', 'lat',
              'lng', 'age', 'maxfloor', 'area', 'landp']
train2, test2 = make_match_df(train, test, MATCH_COLS)

expensive_idx = submission.loc[
    ~test.buildingid.isin(common_ids),
    'pred'].sort_values(ascending=False).index[:10]
bigarea_idx = test.loc[
    ~test.buildingid.isin(common_ids) & ~test.index.isin(expensive_idx),
    'area'].sort_values(ascending=False).index[:20]

REG_COLS = ['pred.v56', 'pred.v59', 'age', 'linecnts', 'floorfromtop']
expensive_pred = []
for idx in expensive_idx:
    res = adaptive_stacking(idx, 20)
    if res is not None:
        expensive_pred.append(res)

REG_COLS = ['pred.v56', 'pred.v59', 'landp',
            'landp_growth', 'maxfloor', 'distance2']
bigarea_pred = []
for idx in bigarea_idx:
    res = adaptive_stacking(idx, 15)
    if res is not None:
        bigarea_pred.append(res)

submission.loc[expensive_idx, 'pred'] = expensive_pred
submission.loc[bigarea_idx, 'pred'] = bigarea_pred
# there is a property with area 1.0, which is supposed to be 10.0. So we fix it here
submission.loc[test.area == 1.0,
               'pred'] = submission.loc[test.area == 1.0, 'pred'] * 10
submission.to_csv(f'final_submission.csv', index=True, header=False)
print('[predict] final submission exported')
