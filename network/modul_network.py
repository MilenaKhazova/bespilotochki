import os
import time
from glob import glob
from tqdm import tqdm
from collections import Counter

import numpy as np

from PIL import ImageDraw, Image, ImageFont
import matplotlib.pyplot as plt
from matplotlib import image as m_image
%matplotlib inline

import cv2

import tensorflow as tf
import keras

from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import DBSCAN

import pytesseract

path = 'Train//'
folder_id = 15  # 0 - 77, no 36
img_id = 10  # other numbers
img = cv2.imread(path + str(folder_id) + "\\" + str(img_id) + ".png")
img.shape
(64, 64, 3)
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_g.shape
(64, 64)
plt.imshow(img_g, cmap='gray');

#Classes
n_classes = 77
letters = 'Ա Բ Գ Դ Ե Զ Է Ը Թ Ժ Ի Լ Խ Ծ Կ Հ Ձ Ղ Ճ Մ Յ Ն Շ Ո Չ Պ Ջ Ռ Ս Վ Տ Ր Ց ՈՒ Փ Ք ԵՎ Օ Ֆ \
           ա բ գ դ ե զ է ը թ ժ ի լ խ ծ կ հ ձ ղ ճ մ յ ն շ ո չ պ ջ ռ ս վ տ ր ց ու փ ք և օ ֆ'

           letters_list = letters.split()
for i in letters_list:
    print(i, end=' ')
Ա Բ Գ Դ Ե Զ Է Ը Թ Ժ Ի Լ Խ Ծ Կ Հ Ձ Ղ Ճ Մ Յ Ն Շ Ո Չ Պ Ջ Ռ Ս Վ Տ Ր Ց ՈՒ Փ Ք ԵՎ Օ Ֆ ա բ գ դ ե զ է ը թ ժ ի լ խ ծ կ հ ձ ղ ճ մ յ ն շ ո չ պ ջ ռ ս վ տ ր ց ու փ ք և օ ֆ
activity_map = {}
for i, j in enumerate(letters_list):
    activity_map[i] = j
del activity_map[36]

# Resizing images size using cv2
def read_resize_image(path, img_rows, img_cols, color_type=1):
    if color_type == 1:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    elif color_type == 3:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (img_rows, img_cols))
    return img
# Train Images
def load_data(img_rows, img_cols, color_type=3):
    start_time = time.time()
    train_images = []
    train_labels = []
    for classed in tqdm(range(n_classes)):
        if classed == 36:
            continue
        print(f'Loading directory {classed}')
        files = glob(os.path.join( 'Train\\' + str(classed), '*.png'))
        for file in files:
            img = read_resize_image(file, img_rows, img_cols, color_type)
            train_images.append(img)
            train_labels.append(classed)

    end_time = time.time()
    print(f"Data Loaded in {(end_time - start_time) // 60} minutes, {round(end_time - start_time) % 60} seconds")

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    return train_images, train_labels
# Normalizing Train Images
def read_norm_split_data(img_rows, img_cols, color_type, test_size=0.2, random_state=42):
    X, labels = load_data(img_rows, img_cols, color_type)

    x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(X, labels, test_size=test_size, random_state=random_state)

    x_train = x_train_0 / 255
    x_test = x_test_0 / 255

    y_train = np_utils.to_categorical(y_train_0, n_classes)
    y_test = np_utils.to_categorical(y_test_0, n_classes)

    x_train = np.array(x_train).reshape(-1, img_rows, img_cols, color_type)
    x_test = np.array(x_test).reshape(-1, img_rows, img_cols, color_type)

    return x_train, x_test, y_train, y_test, y_train_0, y_test_0
img_rows = 64
img_cols = 64
color_type = 1

test_size = 0.2
random_st = 42

x_train, x_test, y_train, y_test, y_train_0, y_test_0 = read_norm_split_data(img_rows, img_cols, color_type, test_size=test_size random_state=random_st)

datagen = ImageDataGenerator(rotation_range=10)#, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(x_train)

def create_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_rows, img_cols, color_type)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(77, activation='softmax'))

    return model
lr_init = 0.001
batch_size = 64
epoch = 100

optimizer_1 = tf.keras.optimizers.Adamax(learning_rate=lr_init)
loss_1 = 'categorical_crossentropy'
metrics_1 = ['accuracy'] #, lr_metric]
model = create_model()
model.summary()
model.compile(optimizer=optimizer_1, loss=loss_1, metrics=metrics_1)

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 30:
        lrate = 0.0005
    if epoch > 60:
        lrate = 0.0003
    if epoch > 80:
        lrate = 0.0001
    return lrate

    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                               steps_per_epoch=x_train.shape[0] // batch_size,
                              epochs=epoch,
#                               verbose=1,
                              validation_data=(x_test, y_test),
                              callbacks=[LearningRateScheduler(lr_schedule)],
                              shuffle=True,
                             )

img = cv2.imread('example_images/2.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray');
ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
plt.imshow(img, cmap='gray');
kernel = np.ones((5, 5), np.int8)
img = cv2.erode(img, kernel, iterations=1)
plt.imshow(img, cmap='gray');

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\user\Milena\tesseract.exe'
text_ = pytesseract.image_to_boxes(img, lang='eng', config=r'--oem 3 --psm 6', output_type='string')
# print(text)
text = text_.split("\n")

im = img.copy()
# h, w, c = img.shape
h, w = img.shape
for b in text_.splitlines():
    b = b.split(' ')
    img_b = cv2.rectangle(im, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 1)

plt.imshow(img_b, cmap='gray');

def change_and_predict_contours(img, thresh=False, img_size=64, model=model_load, activity_map=activity_map, epsilon=4, min_samples=3):
    
    # to grayscale if not
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    # thresholding
    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.int8)
    img = cv2.erode(img, kernel, iterations=1)
    
    if thresh:
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
    
    # resize
    img_64_full = cv2.resize(img, (img_size, img_size))
    img_pad = np.pad(img_64_full, ((30, 30), (30, 30)))
    img_64_ready = cv2.resize(img_pad, (img_size, img_size))
    
    
    ### clustering with DBSCAN
    # generate data
    _, a = cv2.threshold(img_64_ready, 100, 255, cv2.THRESH_BINARY)
    a = a / 255
    data = []
    for i in range(img_size):
        for j in range(img_size):
            if a[i][j]:
                data.append([i, j])
    data = np.array(data)
    
    # clustering and find the biggest cluster
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples)
    p = clustering.fit_predict(data)
    p_c = Counter(p)
    max_kay = 0
    max_value = max(p_c.values())
    for key, value in p_c.items():
        if value == max_value:
            max_key = key
    
    # filter image using cluster
    for i, j in enumerate(p):
        if j != max_key:
            a[data[i][0], data[i][1]] = 0
    
    img_64_final = a
    
    
    # predict
    for_predict = img_64_final.reshape((1, img_size, img_size, 1))
    prob = model.predict(for_predict)
    letter = activity_map[prob.argmax()]
    
    return img_64_final, prob, letter
plt.imshow(img_letters[1]);

letters_final = []
boxes_final = []
probs_final = []

for img_box in img_letters:
    box, prbs, letter = change_and_predict_contours(img_box, thresh=True)
    letters_final.append(letter)
    boxes_final.append(box)
    probs_final.append(prbs)
img_copy = Image.fromarray(img_boxes)

font = ImageFont.truetype("fonts/GHEAGrpalatReg.otf", 64)

for i in range(len(letters_final)):
    img_draw = ImageDraw.Draw(img_copy)
    img_draw.text((boxes[i][0] + 15, boxes[i][1] - 100), text=letters_final[i], fill='green', font=font)
    

