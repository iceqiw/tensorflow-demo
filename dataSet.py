#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

#读取二进制数据
def read_and_decode(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [640, 640, 3])
    adjusted = tf.image.adjust_contrast(img,200)

    rgb_image_float = tf.image.convert_image_dtype(adjusted, tf.float32)
    image_data = tf.image.rgb_to_hsv(rgb_image_float)
    label = tf.cast(label, tf.int32)
    return image_data, label


if __name__ == '__main__':
    
     img, label = read_and_decode("train.tfrecords")
        
     img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=1, capacity=2000,
                                                    min_after_dequeue=1000)
     # 初始化所有的op
     init = tf.initialize_all_variables()
     with tf.Session() as sess:
        sess.run(init)
        # 启动队列
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(2):
            val, l = sess.run([img_batch, label_batch])
            plt.figure(l)
            plt.imshow(val[0])
            plt.show()
    
 
