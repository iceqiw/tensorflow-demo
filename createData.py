#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

cwd = os.getcwd()

classes = {'NG','OK'}


print(cwd)

def create_record():
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd +"/img/"+ name+"/"
        for img_name in os.listdir(class_path):     
            img_path = class_path + img_name
            print(img_path,index)
            img = Image.open(img_path)
            img = img.resize((640, 640))
            img_raw = img.tobytes() #将图片转化为原生bytes
           
            example = tf.train.Example(
               features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
               }))
            writer.write(example.SerializeToString())
    writer.close()



create_record()
