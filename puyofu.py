from tensorflow import keras
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import joblib
from tqdm import tqdm


def score_predict(field_image):
    """
    input : field_image
    output: score [8digits]
    """
    score_model = joblib.load('./model/number_classifier.pkl')

    scores = []

    def crop_score(i, img):
        # クロップする矩形の上辺のY座標
        upper = 885
        lower = 940
        crop_width = 40

        left = 352 + crop_width * i
        right = left + crop_width
        cropped_img = img[upper: lower, left: right]
        return cropped_img

    def process_img(img):
        resized_image = cv2.resize(img, (28, 28))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        image_array = np.array(gray_image)
        image_flattened = (image_array.flatten())
        #     image_reshaped = image_flattened.reshape(1, -1) / 255
        return image_flattened

    def df_maker():
        # 784列のDataFrameを宣言
        num_features = 784
        column_names = ["Feature_" + str(i) for i in range(1, num_features + 1)]
        df = pd.DataFrame(columns=column_names)
        return df

    def is_zero(score):
        return all(num == 0 for num in score)

    for i in range(8):
        tmp_img = crop_score(i, field_image)
        processed_img = process_img(tmp_img)
        # 画像をクロップ
        scores.append(processed_img)

    df = df_maker()
    new_df = pd.DataFrame(scores, columns=df.columns)
    score_predictions = score_model.predict(new_df)

    return score_predictions


def check_image_size(frame):
    # 画像をBGRのカラーチャネル順で読み込む

    height, width, _ = frame.shape

    if height == 1200:
        cropped_frame = frame[60:height - 60, :]
        return cropped_frame
    else:
        return frame

def show_image_list(image_list):
    #field_imagesを一覧表示する
    num_images = len(image_list)
    columns = 6  # 列数
    rows = (num_images + columns - 1) // columns  # 行数

    fig, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes = axes.ravel()  # 1次元化

    for i in range(num_images):
        if i < num_images:
            axes[i].imshow(image_list[i], cmap='gray')  # グレースケールの場合
            axes[i].axis('off')
        else:
            fig.delaxes(axes[i])  # 余分なサブプロットを削除

    fig.set_facecolor('gray')  # プロットの背景色をグレーに設定

    plt.tight_layout()
    plt.show()


def is_blackout(frame):  # 暗転検知
    threshold = 100
    # グレースケールに変換
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ピクセルの平均明るさを計算
    mean_brightness = np.mean(gray_frame)

    # 閾値を下回る場合は暗いフレームとして記録
    if mean_brightness < threshold:
        return True
    else:
        return False


def is_match_start(frame):
    score = score_predict(frame)
    return all(num == 0 for num in score)


def is_nextpuyo_change(prev_frame, current_frame):
    # ネクストぷよ矩形内のRGB差の閾値
    threshold = 5 * 100000

    # ネクストぷよの矩形領域
    next_puyo_frame = dict()
    next_puyo_frame["left"] = 715
    next_puyo_frame["right"] = 785
    next_puyo_frame["upper"] = 160
    next_puyo_frame["bottom"] = 285

    prev = prev_frame[next_puyo_frame["upper"]:next_puyo_frame["bottom"],
           next_puyo_frame["left"]:next_puyo_frame["right"]]
    curr = current_frame[next_puyo_frame["upper"]:next_puyo_frame["bottom"],
           next_puyo_frame["left"]:next_puyo_frame["right"]]

    img_diff = cv2.absdiff(prev, curr).sum()

    if img_diff > threshold:
        return True
    else:
        return False


def is_chain(frame):
    # frameのスコア表示部に❌があればTrue
    score = score_predict(frame)
    if 10 in score:
        return True
    else:
        return False


def skip_10frame(cap):
    for frame_num in range(10):
        ret, frame = cap.read()
        if not ret:
            return False, None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = check_image_size(frame)
    return True, frame


def display_images(image1, image2):
    # 2つの画像を表示
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title('prev')

    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title('curr')

    # 表示
    plt.show()

def main():
    cap = cv2.VideoCapture('../game_captures/match_7.mov')
    state = "暗転"

    ret, frame = cap.read()
    states = []
    field_images = []
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=video_frame_count)
    ct = 1

    while True:
        pbar.update(ct)
        ct = 2
        states.append(state)
        # fpsに合わせる　switchはfps30なので，2回読む
        ret1, frame = cap.read()
        ret2, frame = cap.read()
        if not ret1 or not ret2:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = check_image_size(frame)

        if is_blackout(frame):
            state = "暗転"
            prev_frame = frame
            continue

        if state == "暗転":
            if is_match_start(frame):
                state = "操作"
                prev_frame = frame
            continue

        if state == "操作":
            if is_nextpuyo_change(prev_frame, frame):
                state = "操作"
                field_images.append(prev_frame)
                ret, prev_frame = skip_10frame(cap)
                ct += 10
                if not ret:
                    break
                continue

            if is_chain(frame):
                state = "連鎖"
                ret, prev_frame = skip_10frame(cap)
                ct += 10
                if not ret:
                    break
                continue
            prev_frame = frame
            continue

        if state == "連鎖":
            if is_nextpuyo_change(prev_frame, frame):
                state = "操作"
            prev_frame = frame
            continue

    pbar.close()

if __name__ == "__main__":
    main()
    pass