#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:22:04 2019

@author: happy
"""
import tensorflow as tf
from datetime import datetime

#创建TensorBoard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
 
 个人个人
def relu(X):
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, 0., name="relu")

relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    outvalue=output.eval(feed_dict={X: [[1, 2, 3]]})