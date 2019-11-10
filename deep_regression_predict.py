# -*- coding: utf-8 -*-
import keras
import numpy as np
import pandas as pd
import sys
import os

# 設定ファイルパス
csvfile = 'dataset/train.csv'


def main():

    # 環境設定(ディスプレイの出力先をlocalhostにする)
    os.environ['DISPLAY'] = ':0'

    # 学習データセットファイル取得
    x, y = load_csv(csvfile)
    # 学習モデルファイルパス取得
    savefile = sys.argv[1]
    # 学習済ファイルを読み込んでmodelを作成
    dnn_model = keras.models.load_model(savefile)
    # 入力ベクトルの次元を取得
    n_features = dnn_model.input_shape[1]

    # コマンド引数確認
    if len(sys.argv) < n_features + 2:
        print('使用例: python deep_regression_predict.py モデルファイル名.h5 入力データ')
        sys.exit()

    # 入力データリスト定義
    x_li = []
    for i in range(n_features):
        v_arg = float(sys.argv[(i + 2)])
        x_li.append(v_arg)

    # 入力データベクトル定義
    X = np.array([x_li])
    #print(X)
    # 入力データ標準化
    X = input_normalization(X, x)

    # 予測結果の取得
    result = dnn_model.predict_on_batch(X)

    # 予測結果に対し標準化の逆変換
    Y = result * y.std() + y.mean()
    day = Y[0][0]

    # 結果の表示
    print('ディープラーニングによる予測消化日数')
    print("=" + str(day) + " [日]")


def load_csv(csvfile):
    # csvをロードし、変数に格納
    df = pd.read_csv(csvfile)
    dfv = df.values.astype(np.float64)
    n_dfv = dfv.shape[1]

    # 特徴量のセットを変数Xに、ターゲットを変数yに格納
    x = dfv[:, np.array(range(0, (n_dfv-1)))]
    y = dfv[:, np.array([(n_dfv-1)])]

    return x, y


def input_normalization(X, x):
    # 入力ベクトルの各要素値を、対応する学習データの各列ベクトルで標準化
    n = X.shape[1]
    for i in range(n):
        t = x[:,i]
        X[:,i] = (X[:,i] - t.mean()) / t.std()
    return X


if __name__ == '__main__':
    main()
