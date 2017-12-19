#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()

classes = {'NG','OK'}


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


"""Converts a dataset to TFRecords."""
def convert_to(name,directory):
    filename = os.path.join(directory, name + '.tfrecords')
    with tf.python_io.TFRecordWriter(filename) as record_writer:
        for index, name in enumerate(classes):
            class_path = cwd +"/img/"+ name+"/"
            for img_name in os.listdir(class_path):     
                img_path = class_path + img_name
                print(img_path,index)
                img = Image.open(img_path)
                img = img.resize((640, 640))
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                        'img_raw': _bytes_feature(img_raw),
                        'label': _int64_feature(index)
                }))
                record_writer.write(example.SerializeToString())

if __name__ == '__main__':
    convert_to("train","data")
