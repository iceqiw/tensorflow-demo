#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

HEIGHT = 640
WIDTH = 640
DEPTH = 3


def preprocess(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize_images(image,size=[224,224])
    # image = tf.cast(image, tf.uint8)
    return image


def example_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image.set_shape([WIDTH * HEIGHT * DEPTH])
    label = tf.cast(features['label'], tf.int32)
    image = tf.reshape(image, [640, 640, 3])
    image = preprocess(image)
    return image, tf.one_hot(label, 2)


def train_input_fn(filename, batch_size=1, num_epochs=1):
    dataset = tf.data.TFRecordDataset([filename])

    dataset = dataset.shuffle(buffer_size=108)
    dataset = dataset.repeat(num_epochs)

    dataset = dataset.map(example_parser).prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels

def pred_input_fn(filename):
    image_raw_data_jpg = tf.gfile.FastGFile(filename, 'rb').read()
    img_data = tf.image.decode_jpeg(image_raw_data_jpg, channels=3)
    image = tf.image.rgb_to_grayscale(img_data)
    image = tf.image.resize_images(image,size=[224,224])
    return tf.reshape(image, [-1, 224, 224, 1])

def pred_input_fn2(filename):
  image_raw_data_jpg = tf.gfile.FastGFile(filename, 'rb').read()
  img_data = tf.image.decode_jpeg(image_raw_data_jpg, channels=3)
  image = tf.image.rgb_to_grayscale(img_data)
  image = tf.image.resize_images(image,size=[28,28])
  image=tf.reshape(image, [28,28])
  image = tf.cast(image, tf.float32) / 255 - 0.5
  return image

if __name__ == '__main__':
    img_batch, label_batch = train_input_fn("data/train.tfrecords", 1)
    kk=pred_input_fn("/home/qiwei/workspace/python/tensorflow-demo/wood/img/OK/IMG_3740.jpg")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 启动队列
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(1):
            val, l ,k= sess.run([img_batch, label_batch,kk])
            print(k)
            print(val.shape, k.shape)
            plt.figure(1)
            plt.imshow(val[0,:,:,0], cmap='gray')
            plt.figure(2)
            plt.imshow(k[0,:,:,0], cmap='gray')
            plt.show()
