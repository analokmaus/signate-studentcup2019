import sys
import numpy as np
import pandas as pd
from scipy import stats
from pprint import pprint
from sklearn.linear_model import LinearRegression, Ridge

from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    print('[train] interpolation on progress')
    DATA_PATH = Path('data')
    train = pd.read_csv(DATA_PATH / 'train_complete.csv', index_col=0)
    test = pd.read_csv(DATA_PATH / 'test_complete.csv', index_col=0)

    train_ids = set(train.buildingid.unique())
    test_ids = set(test.buildingid.unique())
    common_ids = train_ids & test_ids

    # Full match
    MATCH_COLS = ['area', 'thisfloor', 'buildingid']
    train_mini = train[['賃料'] +
                       MATCH_COLS].groupby(MATCH_COLS).agg(np.mean).reset_index()
    _test = test[MATCH_COLS]
    _test = _test.merge(train_mini, on=MATCH_COLS, how='left')
    _test.index = test.index

    # Logic
    null_idx = _test['賃料'].isnull().values
    Xy_all = train.loc[train.buildingid.isin(common_ids),
                       ['buildingid', 'thisfloor', 'area', 'floorplan', '賃料']]
    X_test_all = test.loc[test.buildingid.isin(common_ids) & null_idx,
                          ['buildingid', 'thisfloor', 'maxfloor', 'area', 'floorplan']]

    FLOOR_CONST = 1000
    ANOMALY_THRES = 2 * FLOOR_CONST
    worried_idx = []

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

    def remove_outlier(x_train, y, floors=None):
        group = step_search(y, step=10000)
        m = stats.mode(group)[0][0]
        x_train = x_train[group == m, :]
        y = y[group == m]

        if floors is not None:
            floors = floors[group == m]
            return x_train, y, floors
        else:
            return x_train, y

    def simple_regression(x, y, x_test, thres=0.9):
        reg = LinearRegression().fit(x, y)
        score = reg.score(x, y)
        if score > thres:
            y_test = reg.predict(x_test)[0]
            return y_test
        else:
            return None

    for i, (thisId, thisFloor, maxFloor, thisArea, thisFloorplan) in enumerate(tqdm(X_test_all.values)):
        if i == 10000:
            break
        y_test = None
        thisIdx = X_test_all.index[i]

        Xy = Xy_all.loc[Xy_all.buildingid == thisId].values
        match_floors = Xy[:, 1] == thisFloor
        match_areas = abs(Xy[:, 2] / thisArea - 1) < 0.01

        if maxFloor <= 15:
            ANOMALY_THRES = FLOOR_CONST
        else:
            ANOMALY_THRES = 2 * FLOOR_CONST

        if np.sum(match_areas) > 0:

            '''
            Area Match
            '''
            x_train = Xy[match_areas, :-1]
            floors = x_train[:, 1]
            y = Xy[match_areas, -1]

            if len(np.unique(floors)) == 1:  # Same bldg w/ single unique floor: empirical adjustment
                diffFloor = thisFloor - x_train[0, 1]
                if diffFloor > maxFloor / 3:
                    diffFloor = int(maxFloor / 3)
                y_test = np.mean(y)
                y_test += diffFloor * FLOOR_CONST

            else:  # Same bldg w/ multiple unique floor: regression
                # IMP: Detect inconsistency
                if y[np.argmax(floors)] >= y[np.argmin(floors)]:
                    # highest floor should be expensiver than lowest floor
                    # IMP: too big sloop is not good

                    if len(np.unique(floors)) > 3:
                        x_train = floors.reshape(-1, 1)
                        x_test = np.array([thisFloor]).reshape(-1, 1)
                        reg = LinearRegression().fit(x_train, y)
                        if 0 <= reg.coef_[0] <= ANOMALY_THRES:
                            y_test = reg.predict(x_test)[0]

                    if y_test is None:
                        x1 = max(floors)
                        x0 = min(floors)
                        y1 = y[np.argmax(floors)]
                        y0 = y[np.argmin(floors)]
                        slope = (y1 - y0) / (x1 - x0)
                        if slope <= ANOMALY_THRES:
                            # Two point
                            y_test = ((x1 - thisFloor) * y0 +
                                      (thisFloor - x0) * y1) / (x1 - x0)
                        else:
                            # Nearest
                            nearest = np.argmin(abs(thisFloor - floors))
                            diffFloor = thisFloor - floors[nearest]
                            if diffFloor > maxFloor / 3:
                                diffFloor = int(maxFloor / 3)
                            y_test = y[nearest] + diffFloor * ANOMALY_THRES

                else:
                    if max(y) / min(y) > 1.25:
                        # Probably outlier is included
                        x_train, y, floors = remove_outlier(x_train, y, floors)

                    # Nearest
                    nearest = np.argmin(abs(thisFloor - floors))
                    diffFloor = thisFloor - floors[nearest]
                    if diffFloor > maxFloor / 3:
                        diffFloor = int(maxFloor / 3)
                    y_test = y[nearest] + diffFloor * FLOOR_CONST

                    if thisIdx == 62118:
                        print(x_train, y, x_test, y_test)
                        break

        else:

            '''
            No Match
            '''
            areas = Xy[:, 2]
            floors = Xy[:, 1]
            floorplans = Xy[:, 3]
            neighbor = abs(thisArea / areas - 1) < 0.15

            # IMP: Do NOT try to predict completely different sized property
            if np.sum(neighbor) > 0:
                x_train = Xy[neighbor, 2].reshape(-1, 1)
                x_test = np.array([thisArea]).reshape(-1, 1)
                y = Xy[neighbor, -1]
                floors = floors[neighbor]
                areas = areas[neighbor]

                if y_test is None:
                    nearest = np.argmin(abs(thisArea - areas))
                    diffFloor = thisFloor - floors[nearest]
                    if diffFloor > maxFloor / 3:
                        diffFloor = int(maxFloor / 3)
                    y_test = y[nearest] / areas[nearest] * \
                        thisArea + diffFloor * FLOOR_CONST

            else:
                if max(areas) / min(areas) < 1.2:
                    pass
                else:  # multiple point with kind of variance
                    match_fplan = floorplans == thisFloorplan
                    x_train = Xy[:, 2].reshape(-1, 1)
                    x_test = np.array([thisArea]).reshape(-1, 1)
                    y = Xy[:, -1]
                    if np.sum(match_fplan) > 0:
                        x_train = x_train[match_fplan, :]
                        areas = areas[match_fplan]
                        y = y[match_fplan]

                    if y[np.argmax(areas)] >= y[np.argmin(areas)]:
                        reg = LinearRegression().fit(x_train, y)

                        if reg.score(x_train, y) > 0.8:
                            if min(areas) * 0.85 <= thisArea <= max(areas) * 1.15:
                                y_test = reg.predict(x_test)[0]

                            if y_test is not None:
                                floorOffset = thisFloor - max(floors) if thisFloor > max(floors) else \
                                    min(floors) - \
                                    thisFloor if thisFloor < min(floors) else 0
                                y_test += floorOffset * FLOOR_CONST

                    if y_test is None and thisArea >= 70 and np.sum(match_fplan) > 0:
                        nearest = np.argmin(abs(thisArea - areas))
                        if 0.5 < thisArea / areas[nearest] < 1.5:
                            diffFloor = thisFloor - floors[nearest]
                            if diffFloor > maxFloor / 3:
                                diffFloor = int(maxFloor / 3)
                            y_test = y[nearest] / areas[nearest] * \
                                thisArea + diffFloor * FLOOR_CONST
                            worried_idx.append(thisIdx)

        if y_test is not None:
            _test.loc[thisIdx, '賃料'] = y_test
            pass

    print('null', _test['賃料'].isnull().sum())

    _test.loc[~_test['賃料'].isnull(), '賃料'].to_csv(
        DATA_PATH / 'test_fill.csv', index=True, header=False)
