import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

import numpy as np
import pandas as pd

# データの読み込み
train_data = pd.read_csv('_TestData/test_merged.csv', header=None)
train_images = train_data['label']

# train_labelsの型を確認
print(train_images.shape)
# 出力: (データ数, 28*28)


# モデルの定義
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # 入力層
model.add(Dense(128, activation='relu'))  # 中間層1
model.add(Dense(64, activation='relu'))   # 中間層2
model.add(Dense(6, activation='softmax'))  # 出力層 (6クラスの分類)

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの訓練
model.fit(train_images, train_labels, epochs=10, batch_size=32)
