# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats
import sys
import os
import time

# バッチサイズ設定
n_bs = 4
# エポック数
nb_epochs = 1000
# csvファイルパス取得
csvfile = 'dataset/train.csv'

def main():
    # 環境設定(ディスプレイの出力先をlocalhostにする)
    os.environ['DISPLAY'] = ':0'
    # 時間計測開始
    start = time.time()

    # コマンド引数確認
    if len(sys.argv) < 2:
        print('使用法: python deep_regression-20_10.py 保存ファイル名.h5')
        sys.exit()
    # 学習モデルファイルパス取得
    savefile = sys.argv[1]
    # データセットファイル読み込み
    X, y = load_csv(csvfile)

    # サンプル数、特徴量の次元、出力数の取り出し
    (n_samples, n_features) = X.shape
    n_outputs = y.shape[1]

    # モデル作成(既存モデルがある場合は読み込んで再学習。なければ新規作成)
    if os.path.exists(savefile):
            print('モデル再学習')
            dnn_model = keras.models.load_model(savefile)
    else:
            print('モデル新規作成')
            dnn_model = dnn_model_maker(n_features, n_outputs)

    # モデルの学習
    history = dnn_model.fit(X, y, epochs=nb_epochs, validation_split=0.1, batch_size=n_bs, verbose=2)
    # 学習結果を保存
    dnn_model.save(savefile)

    # 学習所要時間の計算、表示
    process_time = (time.time() - start) / 60
    print('process_time = ', process_time, '[min]')

    # 損失関数の時系列変化をグラフ表示
    plot_loss(history)


def load_csv(csvfile):
    # csvをロードし、変数に格納
    df = pd.read_csv(csvfile)
    dfv = df.values.astype(np.float64)
    n_dfv = dfv.shape[1]

    # 特徴量のセットを変数Xに、ターゲットを変数yに格納
    x = dfv[:, np.array(range(0, (n_dfv-1)))]
    y = dfv[:, np.array([(n_dfv-1)])]

    # データの標準化
    x = scipy.stats.zscore(x)
    y = scipy.stats.zscore(y)

    return x, y


def dnn_model_maker(n_features, n_outputs):
    # 4層ニューラルネットワークを定義
    model = Sequential()
    # 中間層1（ニューロン50個）と入力層を定義
    model.add(Dense(units=50, activation='relu', input_shape=(n_features,)))
    # Dropout層を定義
    model.add(Dropout(0.2))
    # 中間層2（ニューロン50個）を定義
    model.add(Dense(units=50, activation='relu'))
    # Dropout層を定義
    model.add(Dropout(0.2))
    # 出力層を定義（ニューロン数は1個）
    model.add(Dense(units=n_outputs, activation='linear'))
    # 回帰学習モデル作成
    model.compile(loss='mean_squared_error', optimizer='adam')
    # モデルを返す
    return model


def plot_loss(history):
    # 損失関数のグラフの軸ラベルを設定
    plt.xlabel('time step')
    plt.ylabel('loss')

    # グラフ縦軸の範囲を0以上と定める
    plt.ylim(0, max(np.r_[history.history['val_loss'], history.history['loss']]))

    # 損失関数の時間変化を描画
    val_loss, = plt.plot(history.history['val_loss'], c='#56B4E9')
    loss, = plt.plot(history.history['loss'], c='#E69F00')

    # グラフの凡例（はんれい）を追加
    plt.legend([loss, val_loss], ['loss', 'val_loss'])

    # 描画したグラフを表示
    #plt.show()

    # グラフを保存
    plt.savefig('dnn_train_figure.png')

if __name__ == '__main__':
    main()
