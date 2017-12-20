#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse
import os
import sys
import tensorflow as tf

from w_input_data import train_input_fn,pred_input_fn

parser = argparse.ArgumentParser()
# Basic model parameters.
parser.add_argument(
    '--batch_size',
    type=int,
    default=10,
    help='Number of images to process in a batch')

parser.add_argument(
    '--data_dir',
    type=str,
    default='data',
    help='Path to the WOOD data directory.')

parser.add_argument(
    '--model_dir',
    type=str,
    default='vgg16_wood_model',
    help='The directory where the model will be stored.')

parser.add_argument(
    '--train_epochs', type=int, default=1, help='Number of epochs to train.')


def wood_model_fn(features, labels, mode, params):
    logits = wood_vgg16_model(features, mode)
   
    predictions = {
        'classes': features,
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    yy = tf.argmax(labels, axis=1)
    tf.identity(yy, name='label')

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    # Configure the training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_op = optimizer.minimize(loss,
                                      tf.train.get_or_create_global_step())
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])

    pyy = predictions['classes']
    tf.identity(pyy, name='prelabel')

    metrics = {'accuracy': accuracy}
    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def wood_vgg16_model(inputs, mode):

    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(inputs, name="conv1_1", kh=3, kw=3, n_out=64)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1_2, pool_size=[2, 2], strides=2)

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2_2, pool_size=[2, 2], strides=2)

    # # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256)
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3_3, pool_size=[2, 2], strides=2)

    # block 4 -- outputs 14x14x512
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512)
    pool4 = tf.layers.max_pooling2d(
        inputs=conv4_3, pool_size=[2, 2], strides=2)

    # block 5 -- outputs 7x7x512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512)
    pool5 = tf.layers.max_pooling2d(
        inputs=conv5_3, pool_size=[2, 2], strides=2)

    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=4096)
    fc6_drop = tf.layers.dropout(
        inputs=fc6,
        name="fc6_drop",
        training=mode == tf.estimator.ModeKeys.TRAIN)

    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096)
    fc7_drop = tf.layers.dropout(
        inputs=fc7,
        name="fc7_drop",
        training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = fc_op(fc7_drop, name="fc8", n_out=2)
    return logits


#定义卷积操作
def conv_op(input_op, name, kh, kw, n_out):
    with tf.name_scope(name) as scope:
        conv = tf.layers.conv2d(
            inputs=input_op,
            filters=n_out,
            kernel_size=[kh, kw],
            padding="same",
            activation=tf.nn.relu,
            name=name)
        return conv


#定义池化层
def mpool_op(input_op, name, kh, kw, s):
    tf.layers.max_pooling2d(
        inputs=input_op, padding='same', pool_size=[2, 2], strides=2)


#定义全连接操作
def fc_op(input_op, name, n_out):
    with tf.name_scope(name) as scope:
        dense = tf.layers.dense(
            inputs=input_op, units=n_out, activation=tf.nn.relu, name=name)
        return dense


def main(unused_argv):
    train_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    wood_classifier = tf.estimator.Estimator(
        model_fn=wood_model_fn, model_dir=FLAGS.model_dir)
    # Set up training hook that logs the training accuracy every 100 steps.
    tensors_to_log = {
        'train_accuracy': 'train_accuracy',
        'label': 'label',
        'prelabel': 'prelabel'
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)

    # Train the model
    wood_classifier.train(
        input_fn=
        lambda: train_input_fn(train_file, FLAGS.batch_size, FLAGS.train_epochs),
        hooks=[logging_hook])
    # predictions = wood_classifier.predict(input_fn=lambda:pred_input_fn("/home/qiwei/workspace/python/tensorflow-demo/wood/img/OK/IMG_3692.jpg"))
    # for i, p in enumerate(predictions):
    #     print("Prediction %s: %s" % (i + 1, p["classes"]))
        


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
