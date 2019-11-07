# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# データセットファイルパス
csvfile = 'dataset/train.csv'


def main():

    # 環境設定(ディスプレイの出力先をlocalhostにする)
    os.environ['DISPLAY'] = ':0'

    # コマンド引数確認
    if len(sys.argv) < 3:
        print('使用法: python deep_regression_plot.py 保存ファイル名.h5 CSVファイル名.csv')
        sys.exit()

    # 学習データセットファイル取得
    x, y = load_csv(csvfile)

    # 学習モデルファイルパス取得
    savefile = sys.argv[1]
    # 学習済ファイルを読み込んでmodelを作成
    dnn_model = keras.models.load_model(savefile)

    # プロット用ファイルパス取得
    plotfile = sys.argv[2]
    # プロット用データロードし変数に格納
    df = pd.read_csv(plotfile)
    dfv = df.values.astype(np.float64)
    n_dfv = dfv.shape[1]

    # 特徴量のセットを変数Xに格納
    X = dfv[:, np.array(range(0, n_dfv))]
    # プロット用のX軸データを別変数に待避
    xp = X.copy()

    # 入力データ標準化
    X = input_normalization(X, x)

    #print("X=")
    #print(X)
    #print("x1=")
    #print(xp)

    # 予測結果の取得
    result = dnn_model.predict(X)

    # 予測結果に対し標準化の逆変換して予測時間（分）を取得
    yp = result[:,0] * y.std() + y.mean()
    #yp = result[:,0]
    print("result=")
    print(yp)

    # 損失関数の時系列変化をグラフ表示
    plt.plot(xp, yp)
    plt.savefig('dnn_xy_figure.png')


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

