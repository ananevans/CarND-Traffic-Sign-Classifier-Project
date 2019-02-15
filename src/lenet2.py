import tensorflow as tf
import math


from tensorflow.contrib.layers import flatten

def LeNet2(x, dropout):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    n_classes = 43
    #weights
    wc1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 108), mean = mu, stddev = sigma))
    wc2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 108, 200), mean = mu, stddev = sigma))
    wd1 = tf.Variable(tf.truncated_normal(shape=(5*5*200 + 14*14*108, 1024), mean = mu, stddev = sigma))
    wd2 = tf.Variable(tf.truncated_normal(shape=(1024, 128), mean = mu, stddev = sigma))
    wout =tf.Variable(tf.truncated_normal(shape=(128, n_classes), mean = mu, stddev = sigma))
    #biases
    bc1 = tf.Variable(tf.zeros(108))
    bc2 = tf.Variable(tf.zeros(200))
    bd1 = tf.Variable(tf.zeros(1024))
    bd2 = tf.Variable(tf.zeros(128))
    bout = tf.Variable(tf.zeros(n_classes))
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x108.
    conv1 = tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, bc1)

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x108. Output = 14x14x108.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x200.
    conv2 = tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, bc2)
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x200. Output = 5x5x200.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # TODO: Flatten. Input = 5x5x200 and 14x14x108..
    conv2 = tf.concat([tf.contrib.layers.flatten(conv2),tf.contrib.layers.flatten(conv1)], axis=1)
    
    # TODO: Layer 3: Fully Connected. Input = . Output = 1024.
    fc1 = tf.add(tf.matmul(conv2, wd1), bd1)
       
    # Activation.
    dropout1 = dropout  # Dropout, probability to keep units
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout1)

    # Layer 4: Fully Connected. Input = 1024. Output = 128.
    fc2 = tf.add(tf.matmul(fc1, wd2), bd2)
    
    # TODO: Activation.
    dropout2 = dropout  # Dropout, probability to keep units
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout2)

    # TODO: Layer 5: Fully Connected. Input = 128. Output = n_classes
    logits = tf.add(tf.matmul(fc2, wout), bout)
    
    return logits
