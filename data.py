import scipy.io
import cv2
from keras.preprocessing import image  
from keras.applications.vgg16 import preprocess_input
import numpy as np   
from keras.utils import np_utils
import tensorflow as tf
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
import random
from keras.layers import Dense, InputLayer, Dropout
from keras.models import model_from_json

def pre_process_image(image, training):
    if training:
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=350,
                                                       target_width=650)
    return image


def training():
    index_val = []
    for x in range(0,5):
        index_val.append(random.randint(1,19))
    X_train = []
    X_valid = []
    for j in range(1, 19):
        image_data_path = '/media/abhishek/New Volume/Kinect Project/data_mapped/sub'+ str(j) + '/'
        for i in range(0, 130):
            w = 10000 + i
            st = str(w)
            file_name = image_data_path + 'MappedFrame' + st[1:] + '.jpg'
            img = cv2.imread(file_name)
            if img is None:
                continue
            else:
                if j in index_val:
                    X_valid.append(img)
                else:
                    X_train.append(img)
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    y_data = pd.read_csv('/media/abhishek/New Volume/Kinect Project/data_mapped/labels.csv')
    y = y_data.label
    y = np_utils.to_categorical(y)
    y_train = []
    y_valid = []
    for j in range(1, 19):
        if j in index_val:
            for i in range((j-1)*129,(j*129)):
                y_valid.append(y[i][:])
        else:
            for i in range((j-1)*129,(j*129)):
                y_train.append(y[i][:])
    print(np.shape(X_train))
    print(np.shape(y_train))
    print(np.shape(X_valid))
    print(np.shape(y_valid))
    X_train = preprocess_input(X_train, mode='tf')
    X_valid = preprocess_input(X_valid, mode='tf')
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    X_train = base_model.predict(X_train)
    X_valid = base_model.predict(X_valid)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    print(X_train.shape)
    X_train = X_train.reshape(1806, 7*7*512)      
    X_valid = X_valid.reshape(516, 7*7*512)
    train = X_train/X_train.max()      
    train = np.array(train)
   
    X_valid = X_valid/X_train.max()
    X_valid = np.array(X_valid)
    model = Sequential()
    model.add(InputLayer((7*7*512,)))    
    model.add(Dense(units=1024, input_shape=(7*7*512,),activation='sigmoid'))
    model.add(Dense(3,input_shape=(1024,), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train, y_train, epochs=20, validation_data=(X_valid, y_valid))
    
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")

def load_weights():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('model.h5')
    print("Loaded model from disk")

    # loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # X = cv2.imread('/media/newhd/Kinect Project/data_mapped/sub1/MappedFrame0001.jpg')
    # X = preprocess_input(X, mode='tf')
    # score = loaded_model.predict(X, verbose=1)
    # print(score)

def main():
    print('Starting main function')
    #load_weights()
    training()
    print('Ending main function')


if __name__ == "__main__":
    main()
