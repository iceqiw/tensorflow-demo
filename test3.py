#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt

image_raw_data_jpg = tf.gfile.FastGFile('/home/qiwei/workspace/python/tensorflow-demo/img/1623008725.jpg', 'rb').read()
# print(image_raw_data_jpg.shape)

with tf.Session() as session:
    img_data = tf.image.decode_jpeg(image_raw_data_jpg, channels=3)
    plt.figure(1)
    plt.imshow(img_data.eval())

    # print(img_data.shape)
    adjusted = tf.image.adjust_contrast(img_data,-1)
    # print(adjusted.shape)
    plt.figure(2)
    plt.imshow(adjusted.eval())

    
    rgb_image_float = tf.image.convert_image_dtype(adjusted, tf.float32)
    image_data = session.run(tf.image.rgb_to_hsv(rgb_image_float))
    # print(image_data.shape)
    plt.figure(3)
    plt.imshow(image_data[:, :, 0], cmap='gray')

    plt.show()
