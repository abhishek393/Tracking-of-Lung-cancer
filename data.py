import scipy.io
import cv2
from keras.preprocessing import image  
import numpy as np   
from keras.utils import np_utils
#from skimage.transform import resize
import tensorflow as tf
import csv

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
    X = []
    for j in range(1, 19):
        image_data_path = '/media/newhd/Kinect Project_new/data_mapped/sub'+ str(j) + '/'
        for i in range(1, 130):
            w = 10000 + i
            st = str(w)
            file_name = image_data_path + 'MappedFrame' + st[1:] + '.jpg'
            img = cv2.imread(file_name)
            if img is None:
                continue
            else:
                print(np.shape(img))
                X.append(img)
    X = np.array(X)
    #dummy_y = np_utils.to_categorical(y)


def main():
    print('main')
    training()
    print('main')


if __name__ == "__main__":
    main()
