from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns

import mojimoji
import re


'''
Data extraction funcs
'''


def parse_address(text, debug=False):
    '''
    Get location
    '''
    text = mojimoji.zen_to_han(text, kana=False)
    text = re.sub(r'[\(（].*[）\)]', '', text)
    text = re.sub(r'[\s、]', '', text)
    _ward = re.search(r'[a-zA-Z一-龥ぁ-んァ-ヶ・ー]+区', text)
    if _ward:
        ward = _ward.group()
        text = re.sub(ward, '', text)  # remove
    else:
        raise ValueError(text)

    text = text.replace('丁目', '-')
    text = text.replace('I', '1')
    text = re.sub(r'(以下|詳細)*未定', '', text)
    text = re.sub(r'[ー‐―−]', '-', text)
    text = re.sub(r'(\d)番[地]*', r'\1-', text)
    text = re.sub(r'[-]{2,}', r'-', text)
    text = re.sub(r'([a-zA-Z一-龥ぁ-んァ-ヶ・ー])(\d)', r'\1-\2', text)
    text = re.sub(r'(\d)号', r'\1', text)
    lines = text.split('-')
    lines += ['' for i in range(4 - len(lines))]  # adjust length
    return (ward, *lines)


def parse_access(text):
    text = text.split("\t\t")
    access_list = []
    for text_i in text:
        text_i = text_i.split('\t')
        if len(text_i) == 3:
            line_name, stat_name, dist = text_i

            # preprocess of line_name
            if "新幹線" in line_name or "ディズニー" in line_name:
                continue

            # preprocess of stat_name
            stat_name = re.sub(r"\(東京メトロ\)", "", stat_name)
            stat_name = re.sub(r"\(.+[県都]\)", "", stat_name)

            # preprocess of dist
            if "バス" in dist:
                # format : '/バス(23分)水道端下車徒歩4分'
                # print(len(text_i), text_i)
                # '/バス(23分)水道端下車徒歩4分' -> 23
                dist = re.sub(r"/バス\((\d+)分\).*", r"\1", dist)
                dist = f"バス{int(dist):02d}分"
                # print(dist)
            elif "車" in dist:
                # format : '車5.2km(18分)'
                # 全て3つ以上あるのでスキップ
                # print(len(text_i), text_i)
                # '車5.2km(18分)' -> 車18分
                dist = re.sub(r"車.+\((\d+)分\)", r"\1", dist)
                dist = f"車{int(dist):02d}分"
                # print(dist)
            else:
                # format : '徒歩10分'
                dist = dist.lstrip("徒歩").rstrip("分")
                dist = int(dist)
                if dist >= 60:
                    dist = "徒歩60>=分"
                else:
                    dist = f"徒歩{dist:02d}分"

            access_list.append((line_name, stat_name, dist))

        else:
            # len(text_i) == 1
            text_i = text_i[0]

            if "分" not in text_i:
                continue

            text_i = text_i.replace("JR", "")
            text_i = text_i.replace("ＪＲ", "")
            text_i = text_i.replace("東京メトロ", "")
            text_i = text_i.replace("京王電鉄", "")
            text_i = text_i.replace("東京都", "")
            text_i = text_i.replace("メトロ", "")

            text_i = re.sub(r"[\.『』「」：。・、\u3000]", "", text_i)  # 記号を取り除く

            text_i = text_i.replace("丸の内", "丸ノ内")
            text_i = text_i.replace("井の頭", "井ノ頭")
            text_i = text_i.replace("京王井ノ頭", "井ノ頭")

            if "バス" in text_i:
                if len(re.findall("分", text_i)) == 1:
                    # '都営バス／堀ノ内\u30002分' etc...
                    continue
                else:
                    # format : '山手線渋谷駅バス8分池尻住宅前下車徒歩5分'
                    # '分'が２回出てきてる
                    line_name = re.sub(r"(.+線).+", r"\1", text_i)
                    stat_name = re.sub(r".+線(.+駅).+", r"\1", text_i)
                    dist = re.sub(r".+バス(\d+)分.+", r"\1", text_i)
                    dist = f"バス{int(dist):02d}分"
                    # print(line_name, stat_name, dist)
                    access_list.append((line_name, stat_name, dist))
            else:
                # 以降全て徒歩分
                if "線" not in text_i and "駅" not in text_i:
                    # '小72／王子製紙裏門　4分'
                    # '東急・小田急「大橋停留所」停徒歩3分'
                    # etc
                    continue

                text_i = re.sub(r"分(.)", r"分\t\1", text_i)
                text_i = text_i.split('\t')

                for text_i_j in text_i:
                    if "m" in text_i_j or "ⅿ" in text_i_j:
                        # 720m的なやつ
                        continue
                    text_i_j = text_i_j.replace("/", "")
                    text_i_j = text_i_j.replace("徒歩", "")

                    if "駅" not in text_i_j:
                        # print(text_i_j, text_i)
                        line_name = re.sub(r"([^線]+線).+", r"\1", text_i_j)
                        stat_name = re.sub(
                            r".+線([^\d]+)\d+分", r"\1", text_i_j) + "駅"
                        dist = re.sub(r"[^\d]+(\d+)分.*", r"\1", text_i_j)
                        # print(line_name, stat_name, dist)
                    elif "線" not in text_i_j:
                        _match = re.search(
                            r"東京モノレール|ゆりかもめ|東京臨海高速鉄道|日暮里舎人ライナー", text_i_j)
                        if not _match:
                            continue
                        line_name = re.sub(
                            r"(東京モノレール|ゆりかもめ|東京臨海高速鉄道|日暮里舎人ライナー).*", r"\1", text_i_j)
                        stat_name = re.sub(
                            r"(東京モノレール|ゆりかもめ|東京臨海高速鉄道|日暮里舎人ライナー)([^\d]*)\d+分", r"\2", text_i_j)
                        dist = re.sub(
                            r"(東京モノレール|ゆりかもめ|東京臨海高速鉄道|日暮里舎人ライナー)[^\d]*(\d+)分", r"\2", text_i_j)
                        # print(line_name, stat_name, dist)
                    else:
                        line_name = re.sub(r"([^線]+線).+", r"\1", text_i_j)
                        stat_name = re.sub(r".+線(.+駅).+", r"\1", text_i_j)
                        dist = re.sub(r"[^\d]+(\d+)分.*", r"\1", text_i_j)
                        # print(line_name, stat_name, dist, text_i_j)

                    dist = int(dist)
                    if dist >= 60:
                        dist = "徒歩60>=分"
                    else:
                        dist = f"徒歩{dist:02d}分"
                    # print(line_name, stat_name, dist, text_i_j, sep="   \t")

                    access_list.append((line_name, stat_name, dist))

    for i in range(len(access_list)):
        line_name, stat_name, dist = access_list[i]

        # line_name
        if line_name.startswith("湘南新宿ライン"):
            line_name = "湘南新宿ライン"
        elif line_name.startswith("中央本線"):
            line_name = "中央本線"
        elif line_name == "東武伊勢崎大師線":
            line_name = "東武スカイツリーライン"
        elif line_name == "東武伊勢崎線":
            line_name = "東武伊勢崎線(押上－曳舟)"
        elif line_name == "京王井の頭線":
            line_name = "井ノ頭線"
        elif line_name == "京成電鉄本線":
            line_name = "京成本線"
        elif line_name == "三田線":
            line_name = "都営三田線"
        elif line_name in ["京王小田急線", "小田急電鉄小田原線"]:
            line_name = "小田急小田原線"
        elif line_name in ["中央総武線", "中央総武緩行線"]:
            line_name = "総武線・中央線（各停）"
        elif line_name in ["地下鉄浅草線", "浅草線"]:
            line_name = "都営浅草線"
        elif line_name == "西武池袋豊島線":
            line_name = "西武池袋線"
        elif line_name == "総武線":
            line_name = "総武線・中央線（各停）"
        elif line_name == "東京臨海高速鉄道":
            line_name = "りんかい線"
        elif line_name == "東京モノレール":
            line_name = "東京モノレール羽田線"
        elif line_name == "日暮里舎人ライナー":
            line_name = "日暮里・舎人ライナー"
        elif line_name == "大井町線":
            line_name = "東急大井町線"
        elif line_name == "千代田常磐緩行線":
            line_name = "常磐線"
        elif line_name == "中央線":
            line_name = "中央線（快速）"
        elif line_name == "丸ノ内線":
            line_name = "丸ノ内線(池袋－荻窪)"
        elif line_name == "京成成田空港線":
            line_name = "京成本線"
        elif line_name == "京浜東北根岸線":
            line_name = "京浜東北線"
#             print(line_name, stat_name, text)

        access_list[i] = [line_name, stat_name, dist]

    return np.array(access_list)


def encode_access(text):
    '''
    Returns
    toho / bus / car from nearest station
    line
    moyori station
    moyori mins
    '''
    access = parse_access(text)
    dists = access[:, 2]
    how = np.ones(access.shape[0], dtype=np.uint8)
    # 1: toho/ 4: bus/ 5: car
    for i, d in enumerate(dists):
        if 'バス' in d:
            how[i] = 4
        elif '車' in d:
            how[i] = 5
        d = int(re.search(r'\d+', d).group())
        dists[i] = d * how[i]

    access = access[np.argsort(dists.astype(int))]
    if access.shape[0] < 3:
        _access = np.full((3 - access.shape[0], 3), '',)
        access = np.vstack((access, _access))

    flat_a = access[:3].flatten()

    return flat_a


def parse_age(text, debug=False):
    '''
    Get property age
    '''
    if text == '新築':
        return 0.0
    _year = re.search(r'\d+年', text)
    _month = re.search(r'\d+ヶ月', text)
    if _year:
        year = int(_year.group()[:-1])
    else:
        return np.nan
    if _month:
        month = int(_month.group()[:-2])
    else:
        month = 0

    age = year + month / 12
    return age


def parse_area(text, debug=False):
    '''
    Get area
    '''
    area = float(text[:-2])
    return area


def parse_floor(text, debug=False):
    if text != text:
        return np.nan, np.nan
    _thisfloor = re.search(r'\d+階', text)
    _maxfloor = re.search(r'\d+階建', text)
    if _thisfloor:
        thisfloor = int(_thisfloor.group()[:-1])
    else:
        thisfloor = np.nan
    if _maxfloor:
        maxfloor = int(_maxfloor.group()[:-2])
    else:
        maxfloor = np.nan

    return thisfloor, maxfloor


def parse_etc(text, debug=False):
    '''
    Get bath, kitchen, etc...
    '''
    if text != text:
        return set()
    text = mojimoji.zen_to_han(text, kana=False)
    text = text.replace('/', '')
    vals = set(text.split('\t'))
    vals -= {''}
    return vals


def get_union(arr, debug=False):
    '''
    Get value sets
    '''
    all_vals = set()
    for val in arr:
        all_vals = all_vals.union(parse_etc(val))
    return all_vals


def encode_etc(text, valset, debug=False):
    '''
    Encode bath, kitchen, etc...
    '''
    valset = np.array(valset)
    vals = np.array(list(parse_etc(text, debug)))
    if len(vals) == 0:
        return np.full(len(valset), 9, dtype=np.uint8)
    if debug:
        print(valset, vals)
    return np.isin(valset, vals).astype(np.uint8)


def parse_env(text, debug=False):
    if text != text:
        return set()
    text = mojimoji.zen_to_han(text, kana=False)
    return set(re.findall(r'【.{1,9}】', text))


def get_union2(arr, debug=False):
    all_vals = set()
    for val in arr:
        all_vals = all_vals.union(parse_env(val))
    return all_vals


def encode_env(text, valset, debug=False):
    '''
    Encode environment
    '''
    items = parse_etc(text, debug)
    if len(items) == 0:
        return np.full(len(valset), 9999)
    dists = np.full(len(valset), 2048)
    for item in items:
        place = re.search(r'【.{1,9}】', item).group()
        idx = valset.index(place)
        dist = int(re.search(r'\d+', item).group())
        if dist < dists[idx]:
            dists[idx] = dist
    return dists


def parse_park(text, debug=False):
    if text != text:
        return np.full(3, 9, dtype=np.uint8)
    text = mojimoji.zen_to_han(text, kana=False)
    text = text.replace('\t', '')
    keys = re.findall(r'駐輪場|バイク置き場|駐車場', text)
    vals = re.split(r'駐輪場|バイク置き場|駐車場', text)[1:]
    assert len(keys) == len(vals)
    res = np.zeros(3, dtype=np.uint8)
    key_order = ['駐輪場', 'バイク置き場', '駐車場']
    for i, v in enumerate(vals):
        if '無' in v:
            res[key_order.index(keys[i])] = 0
        else:
            _price = re.search(r'\d+円', v)
            if _price:
                res[key_order.index(keys[i])] = 2
            elif '空有' in v:
                res[key_order.index(keys[i])] = 1

    return res


def parse_contr(text, debug=False):
    if text != text:
        return np.nan
    text = mojimoji.zen_to_han(text, kana=False)
    _yearmonth = re.search(r'(\d+年\d+ヶ月間|\d+年間|\d+ヶ月間)', text)
    if _yearmonth:
        ym = re.findall(r'\d+', _yearmonth.group())
        if len(ym) == 2:
            year, month = ym
        elif _yearmonth.group()[-2] == '年':
            year, month = ym[0], 0
        elif _yearmonth.group()[-2] == '月':
            year, month = 0, ym[0]
        left_year = int(year) + int(month) / 12
        return left_year
    else:
        _untilym = re.search(r'\d+年\d+月まで', text)
        if _untilym:
            year, month = re.findall(r'\d+', _untilym.group())
            left_year = int(year) - 2019 + (int(month) - 7) / 12
            return left_year


def parse_plan(text, debug=False):
    if '+S(納戸)' in text:
        text = text.replace('+S(納戸)', '')
        return text, 1
    else:
        return text, 0


'''
Pipeline
'''


def preprocess_df(df, drop=False):
    df2 = df.copy()

    # Add address
    vals = df['所在地'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append(parse_address(val))
    new_df = pd.DataFrame(new_df, index=df.index, columns=[
                          f'adr{i}' for i in range(5)])
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('所在地', axis=1)

    # Add access
    vals = df['アクセス'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append(encode_access(val))
    new_df = pd.DataFrame(new_df, index=df.index,
                          columns=[f'{key}{i}' for i in range(3) for key in ['line', 'station', 'distance']])
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('アクセス', axis=1)

    # Add age
    vals = df['築年数'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append([parse_age(val)])
    new_df = pd.DataFrame(new_df, index=df.index, columns=['age'])
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('築年数', axis=1)

    # Add area
    vals = df['面積'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append([parse_area(val)])
    new_df = pd.DataFrame(new_df, index=df.index, columns=['area'])
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('面積', axis=1)

    # Add floor plan
    vals = df['間取り'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append(parse_plan(val))
    new_df = pd.DataFrame(new_df, index=df.index, columns=[
                          'floorplan', 'withstorage'])
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('間取り', axis=1)

    # Add floor
    vals = df['所在階'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append(parse_floor(val))
    new_df = pd.DataFrame(new_df, index=df.index, columns=[
                          'thisfloor', 'maxfloor'])
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('所在階', axis=1)

    # Add bath, toillet
    valset = {'トイレなし', '共同トイレ', '専用バス', '脱衣所', 'バス・トイレ別',
              '専用トイレ', 'シャワー', '洗面台独立', '共同バス', '浴室乾燥機',
              '追焚機能', '温水洗浄便座', 'バスなし'}
    valset = sorted(valset)
    vals = df['バス・トイレ'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append(encode_etc(val, valset))
    new_df = pd.DataFrame(new_df, index=df.index, columns=list(valset))
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('バス・トイレ', axis=1)

    # Add kitchen
    valset = {'ガスコンロ', '冷蔵庫あり', 'コンロ設置可(コンロ4口以上)', 'カウンターキッチン',
              'IHコンロ', 'コンロ設置可(コンロ1口)', 'コンロ設置可(コンロ3口)', 'コンロ3口',
              '電気コンロ', 'コンロ設置可(口数不明)', 'L字キッチン', '独立キッチン', '給湯',
              'コンロ4口以上', 'コンロ2口', 'システムキッチン', 'コンロ1口', 'コンロ設置可(コンロ2口)'}
    valset = sorted(valset)
    vals = df['キッチン'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append(encode_etc(val, valset))
    new_df = pd.DataFrame(new_df, index=df.index, columns=list(valset))
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('キッチン', axis=1)

    # Add tv
    valset = {'BSアンテナ', '光ファイバー', 'CSアンテナ', '高速インターネット', 'CATV',
              '有線放送', 'インターネット対応', 'インターネット使用料無料'}
    valset = sorted(valset)
    vals = df['放送・通信'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append(encode_etc(val, valset))
    new_df = pd.DataFrame(new_df, index=df.index, columns=list(valset))
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('放送・通信', axis=1)

    # Add interior
    valset = {'バルコニー', '井戸', 'ルーフバルコニー', 'ガス暖房', '室内洗濯機置場',
              '汲み取り', 'シューズボックス', 'エレベーター', '室外洗濯機置場', 'エアコン付',
              'バリアフリー', 'プロパンガス', '浄化槽', '床暖房', 'ロフト付き', 'フローリング',
              '専用庭', '石油暖房', '3面採光', 'クッションフロア', '公営水道', 'ガスその他',
              '出窓', '冷房', '水道その他', '防音室', '都市ガス', '下水', 'オール電化',
              '二世帯住宅', 'ウォークインクローゼット', '敷地内ごみ置き場', 'トランクルーム',
              '24時間換気システム', '排水その他', 'タイル張り', '2面採光', '二重サッシ',
              '床下収納', '地下室', '洗濯機置場なし', 'ペアガラス'}
    valset = sorted(valset)
    vals = df['室内設備'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append(encode_etc(val, valset))
    new_df = pd.DataFrame(new_df, index=df.index, columns=list(valset))
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('室内設備', axis=1)

    # Add environment
    valset = {'【銀行】', '【コンビニ】', '【小学校】', '【総合病院】', '【大学】',
              '【図書館】', '【デパート】', '【公園】', '【幼稚園・保育園】',
              '【コインパーキング】', '【クリーニング】', '【レンタルビデオ】',
              '【学校】', '【ドラッグストア】', '【月極駐車場】', '【スーパー】',
              '【病院】', '【飲食店】', '【郵便局】'}
    valset = sorted(valset)
    vals = df['周辺環境'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append(encode_env(val, valset))
    new_df = pd.DataFrame(new_df, index=df.index, columns=list(valset))
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('周辺環境', axis=1)

    # Add parking
    vals = df['駐車場'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append(parse_park(val))
    new_df = pd.DataFrame(new_df, index=df.index,
                          columns=['park_bike', 'park_motor', 'park_car'])
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('駐車場', axis=1)

    # Add contract
    vals = df['契約期間'].values
    new_df = []
    for i, val in enumerate(vals):
        new_df.append([parse_contr(val)])
    new_df = pd.DataFrame(new_df, index=df.index, columns=['contr'])
    df2 = df2.merge(new_df, left_index=True, right_index=True)
    if drop:
        df2 = df2.drop('契約期間', axis=1)

    return df2


'''
Fix typo
'''

train_fix = [
    [3335, '所在地', '東京都中央区晴海２丁目２－２－４２', '東京都中央区晴海２丁目２－４２'],
    [5776, '賃料', 1203500, 123500],
    [7089, '所在地', '東京都大田区池上８丁目8-6-2', '東京都大田区池上８丁目6-2'],
    [7492, '面積', '5.83m2', '58.3m2'],
    [9483, '所在地', '東京都世田谷区太子堂一丁目', '東京都世田谷区太子堂1丁目'],
    [19366, '所在地', '東京都大田区池上８丁目8-6-2', '東京都大田区池上８丁目6-2'],
    [20232, '築年数', '520年5ヶ月', '20年5ヶ月'],
    [20428, '築年数', '1019年7ヶ月', '19年7ヶ月'],
    [20888, '所在地', '東京都大田区本羽田一丁目', '東京都大田区本羽田1丁目'],
    [20927, '面積', '430.1m2', '43.1m2'],
    [21286, '所在地', '東京都北区西ケ原３丁目西ヶ原３丁目', '東京都北区西ケ原３丁目'],
    [22250, '所在地', '東京都中央区晴海２丁目２－２－４２', '東京都中央区晴海２丁目２－４２'],
    [22884, '所在地', '東京都新宿区下落合２丁目2-1-17', '東京都新宿区下落合２丁目1-17'],
    [27199, '所在地', '東京都中央区晴海２丁目２－２－４２', '東京都中央区晴海２丁目２－４２'],
    [28141, '所在地', '東京都北区西ケ原１丁目西ヶ原１丁目', '東京都北区西ケ原１丁目']
]

test_fix = [
    [34519, '所在地', '東京都足立区梅田１丁目1-8-16', '東京都足立区梅田１丁目8-16'],
    [34625, '所在地', '東京都渋谷区千駄ヶ谷３丁目3-41-12', '東京都渋谷区千駄ヶ谷３丁目41-12'],
    [36275, '所在地', '東京都大田区本羽田一丁目', '東京都大田区本羽田1丁目'],
    [40439, '所在地', '東京都品川区東品川四丁目', '東京都品川区東品川4丁目'],
    [41913, '所在地', '東京都板橋区志村１丁目１－８－１', '東京都板橋区志村１丁目８－１'],
    [45863, '所在地', '東京都大田区東糀谷３丁目3-2-2', '東京都大田区東糀谷３丁目2-2'],
    [49887, '所在地', '東京都大田区大森北一丁目', '東京都大田区大森北1丁目'],
    [56202, '所在地', '東京都大田区新蒲田３丁目9--20', '東京都大田区新蒲田３丁目9-20'],
    [57445, '所在地', '東京都目黒区八雲二丁目', '東京都目黒区八雲2丁目'],
    [58136, '所在地', '東京都文京区本駒込６丁目１－２２－４０３', '東京都文京区本駒込６丁目１－２２'],
    [58987, '所在地', '東京都北区西ケ原４丁目西ヶ原４丁目', '東京都北区西ケ原４丁目']
]


if __name__ == "__main__":
    DATA_PATH = Path('data')

    train = pd.read_csv(DATA_PATH / 'train.csv', index_col=0)
    test = pd.read_csv(DATA_PATH / 'test.csv', index_col=0)

    print('[preprocess] fixing typo')
    for idx, col, prev, new in train_fix:
        # print(idx, col, train.loc[idx, col], '->', new)
        train.loc[idx, col] = new

    for idx, col, prev, new in test_fix:
        # print(idx, col, test.loc[idx, col], '->', new)
        test.loc[idx, col] = new

    print('[preprocess] parsing data')
    train2 = preprocess_df(train, True)
    test2 = preprocess_df(test, True)

    train2.to_csv(DATA_PATH / 'train_processed.csv')
    test2.to_csv(DATA_PATH / 'test_processed.csv')
