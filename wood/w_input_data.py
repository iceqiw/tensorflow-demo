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
    # Pad 4 pixels on each dimension of feature map, done in mini-batch
    image = tf.image.resize_images(image,size=[28,28])
    
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
    image=preprocess(image)
    return image, tf.one_hot(label, 2)

def input_fn(filename, batch_size=1, num_epochs=1):
    dataset = tf.data.TFRecordDataset([filename])
    dataset = dataset.repeat(num_epochs)

    dataset = dataset.map(example_parser).prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    return images, labels


if __name__ == '__main__':

    img_batch, label_batch = input_fn("data/train.tfrecords", 1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 启动队列
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(2):
            val, l = sess.run([img_batch, label_batch])
            print(val.shape)
            print(l.shape, l)
            plt.figure(1)
            plt.imshow(val[0])
            plt.show()
