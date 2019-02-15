import load_data
import tensorflow as tf
import lenet2
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt

def grayscale(images):
    new_images = np.zeros((images.shape[0], images.shape[1], images.shape[2]) )
    for i in range(images.shape[0]):
        new_images[i] = cv.cvtColor(images[i], cv.COLOR_RGB2GRAY)
    return new_images

import csv

label2Name = {}
with open('../signnames.csv') as namesFile:
    nameReader = csv.reader(namesFile)
    for row in nameReader:
        label2Name[int(row[0])] = row[1]

tf.Session(config=tf.ConfigProto(log_device_placement=True))

EPOCHS = 40
BATCH_SIZE = 256 

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples



names = glob.glob('../test-images/*.jpg')
images = [ plt.imread('./' + name ) for name in names ]

# convert to grayscale
print(np.array(images).shape)
X_data = grayscale(np.array(images))
X_data = np.reshape(X_data, (X_data.shape[0],X_data.shape[1],X_data.shape[2], 1))
X_data = (X_data-128)/128

n_classes =  43

# Shuffle the training data.
from sklearn.utils import shuffle

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

logits = lenet2.LeNet2(x, 1.0)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
#optimizer = tf.train.MomentumOptimizer(rate, 0.9, use_nesterov=True)

training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./lenet2.meta')
    saver2.restore(sess, './lenet2')
    y = sess.run(logits, feed_dict={x: X_data})
    labels = sess.run(tf.argmax(y,1))
    #print(labels)
    for i in range(len(labels)):
        print('Label for file', names[i], 'is', label2Name[labels[i]])
    top_prob = sess.run(tf.nn.top_k(tf.nn.softmax(y),3))
    for i in range(len(labels)):
        print('For file', names[i], ' top 3 are:')
        for j in range(3):
             print('%2.4f' % (top_prob[0][i][j]), 'for', label2Name[top_prob[1][i][j]])

