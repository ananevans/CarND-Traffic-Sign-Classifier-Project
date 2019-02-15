import tensorflow as tf
import math


from tensorflow.contrib.layers import flatten

def LeNet(x,dropout):    
    n_classes = 43
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    # Store layers weight & bias
    wc1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    wc2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    wd1 = tf.Variable(tf.truncated_normal(shape=(5*5*16 + 14*14*6, 512), mean = mu, stddev = sigma))
    wd2 = tf.Variable(tf.truncated_normal(shape=(512, 128), mean = mu, stddev = sigma))
    wout = tf.Variable(tf.truncated_normal(shape=(128, n_classes), mean = mu, stddev = sigma))

    # biases 
    bc1 = tf.Variable(tf.zeros(6))
    bc2 = tf.Variable(tf.zeros(16))
    bd1 = tf.Variable(tf.zeros(512))
    bd2 = tf.Variable(tf.zeros(128))
    bout = tf.Variable(tf.zeros(n_classes))
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x16.
    conv1 = tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, bc1)

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x16. Output = 14x14x16.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
   
    l1_fwd = tf.contrib.layers.flatten(conv1)
 
    # Layer 2: Convolutional. Output = 10x10x32.
    conv2 = tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, bc2)
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x32. Output = 5x5x32.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Flatten. Input = 5x5x32.
    conv2 = tf.concat( [tf.contrib.layers.flatten(conv2), l1_fwd], 1)
    
    # Layer 3: Fully Connected. Input = 1600. Output = 512.
    fc1 = tf.add(tf.matmul(conv2, wd1), bd1)
       
    # Activation.
    dropout1 = dropout  # Dropout, probability to keep units
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout1)

    # Layer 4: Fully Connected. Input = 512. Output = 128.
    fc2 = tf.add(tf.matmul(fc1, wd2), bd2)
    
    # TODO: Activation.
    dropout2 = dropout  # Dropout, probability to keep units
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout2)

    # TODO: Layer 5: Fully Connected. Input = 128. Output = n_classes
    logits = tf.add(tf.matmul(fc2, wout), bout)
    
    return logits
