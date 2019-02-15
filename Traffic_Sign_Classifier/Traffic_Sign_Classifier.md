
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data

First, I augment the training data set with the following operations: apply histogram equalization, apply CLAHE algorithm, rotation with random angle between -15 and 15 degrees, translation, shearing, sher followed by rotation, as well as rotation of images with increased contrast.


```python
import pickle
import cv2 as cv
import numpy as np
import random
import math

# Augment the training data

def grayscale(images):
    new_images = np.zeros((images.shape[0], images.shape[1], images.shape[2]) )
    for i in range(images.shape[0]):
        new_images[i] = cv.cvtColor(images[i], cv.COLOR_RGB2GRAY)
    return new_images

def equalizeHist(images):
    new_images = np.zeros((images.shape[0], images.shape[1], images.shape[2]) )
    for i in range(images.shape[0]):
        img = cv.cvtColor(images[i], cv.COLOR_RGB2GRAY)
        new_images[i] = cv.equalizeHist(img)
    return new_images

def clahe(images):
    new_images = np.zeros((images.shape[0], images.shape[1], images.shape[2]) )
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for i in range(images.shape[0]):
        img = cv.cvtColor(images[i], cv.COLOR_RGB2GRAY)
        new_images[i] = clahe.apply(img)
    return new_images

def rotation(images):
    new_images = np.copy(images)
    for i in range(images.shape[0]):
        # generate random angle between -15 and 15 degrees
        theta = random.uniform(-math.radians(15), math.radians(15))
        rows,cols = images[i].shape
        M = cv.getRotationMatrix2D((cols/2,rows/2),theta,1)
        new_images[i] = cv.warpAffine(images[i],M,(cols,rows))
    return new_images
                           
def translation(images):
    new_images = np.copy(images)
    for i in range(images.shape[0]):
        # generate random translation params
        delta_x = random.randint(-2,2)
        delta_y = random.randint(-2,2)
        rows,cols = images[i].shape
        M = np.float32([[1,0,delta_x],[0,1,delta_y]])
        new_images[i] = cv.warpAffine(images[i],M,(cols,rows))
    return new_images

def scale(images):
    new_images = np.copy(images)
    for i in range(images.shape[0]):
        factor = random.uniform(0.9,1.1)
        new_images[i] = cv.resize(images[i],None,fx=factor, fy=factor, interpolation = cv.INTER_CUBIC)
    return new_images


def shear(images):
    new_images = np.copy(images)
    for i in range(images.shape[0]):
        src = np.float32([[5,5],[5,20],[20,5]])
        dst = np.float32([[5 + random.randint(-2,2), 5 + random.randint(-2,2)],
                          [5 + random.randint(-2,2), 20 + random.randint(-2,2)],
                          [20 + random.randint(-2,2), 5  + random.randint(-2,2)]])
        rows,cols = images[i].shape
        M = cv.getAffineTransform(src,dst)
        new_images[i] = cv.warpAffine(images[i],M,(cols,rows))
    return new_images  
```


```python
# load file
training_file = '/home/ans5k/work/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/train.p'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X_train, y_train = train['features'], train['labels']

# convert to grayscale
gray_images = grayscale(X_train)
all_images = np.copy(gray_images)
all_y = np.copy(y_train)
# improve contrast
all_images = np.concatenate((all_images, equalizeHist(X_train)))
all_y  = np.concatenate((all_y, y_train))
# clahe
all_images = np.concatenate((all_images, clahe(X_train)))
all_y  = np.concatenate((all_y, y_train))
# rotation
all_images =  np.concatenate((all_images, rotation(gray_images)))
all_y  = np.concatenate((all_y, y_train))
all_images =  np.concatenate((all_images, rotation(gray_images)))
all_y  = np.concatenate((all_y, y_train))
# translation
all_images =  np.concatenate((all_images, translation(gray_images)))
all_y  = np.concatenate((all_y, y_train))
all_images =  np.concatenate((all_images, translation(gray_images)))
all_y  = np.concatenate((all_y, y_train))
# shear
all_images =  np.concatenate((all_images, shear(gray_images)))
all_y  = np.concatenate((all_y, y_train))
all_images =  np.concatenate((all_images, shear(gray_images)))
all_y  = np.concatenate((all_y, y_train))

all_images =  np.concatenate((all_images, rotation(shear(gray_images))))
all_y  = np.concatenate((all_y, y_train))
all_images =  np.concatenate((all_images, rotation(clahe(X_train))))
all_y  = np.concatenate((all_y, y_train))
all_images =  np.concatenate((all_images, rotation(equalizeHist(X_train))))
all_y  = np.concatenate((all_y, y_train))

augmented_training_file = '/home/ans5k/work/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/augmented-train.p'
with open(augmented_training_file, mode = 'wb') as f:
    pickle.dump({'features': all_images, 'labels': all_y}, f)
```

Convert to grayscale the validation and test images.


```python
validation_file = 'traffic-signs-data/valid.p'
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)

X_valid, y_valid = valid['features'], valid['labels']
    
with open('traffic-signs-data/gray-valid.p', mode = 'wb') as f:
    pickle.dump({'features': grayscale(X_valid), 'labels': y_valid}, f)

```


```python
testing_file = 'traffic-signs-data/test.p'
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
X_test, y_test = test['features'], test['labels']
with open('traffic-signs-data/gray-test.p', mode = 'wb') as f:
    pickle.dump({'features': grayscale(X_test), 'labels': y_test}, f)
```

Run the cell below to load the new pickle files with augmented and preprocessed data:


```python
# Load pickled data
import numpy as np
import pickle

training_file = 'traffic-signs-data/augmented-train.p'
validation_file = 'traffic-signs-data/gray-valid.p'
testing_file = 'traffic-signs-data/gray-test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],X_train.shape[2], 1))
X_valid = np.reshape(X_valid, (X_valid.shape[0],X_valid.shape[1],X_valid.shape[2], 1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],X_test.shape[2], 1))
```

---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = max(y_train) + 1

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

```

    Number of training examples = 417588
    Number of validation examples = 4410
    Number of testing examples = 12630
    Image data shape = (32, 32, 1)
    Number of classes = 43


### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline

import random
import numpy as np

import csv

label2Name = {}
with open('signnames.csv') as namesFile:
    nameReader = csv.reader(namesFile)
    for row in nameReader:
        label2Name[int(row[0])] = row[1]

fig, axes = plt.subplots(5, 5, figsize=(10,10),
                         subplot_kw={'xticks': [], 'yticks': []})
   
indexes = list(random.randint(0, n_train-1) for r in range(25))
labelLimit = 25
for ax, index in zip(axes.flat, indexes):
    ax.imshow(np.reshape(X_train[index],(32,32)), cmap='gray')

```


![png](output_15_0.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

import numpy as np
import cv2

# data is already in grayscale, performing normalization

X_train = (X_train-128)/128
X_valid = (X_valid-128)/128
X_test = (X_test-128)/128

```

### Architecture

The first of the architecture I tried is based on the one presented in class. It has two convolution layers with depths 6 and 16 and max poooling with kernel size 2x2. I added dropout regularization with rate 0.8. As in the Sermanet and LeCun paper, I added the output of the first convolutional layer to the input of the first fully conected layer.


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
import math
from tensorflow.contrib.layers import flatten

def LeNet(x, dropout):    
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
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1 = tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, bc1)

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
   
    l1_fwd = tf.contrib.layers.flatten(conv1)
 
    # Layer 2: Convolutional. Output = 10x10x16.
    conv2 = tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, bc2)
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Flatten. Output = 5*5*16 + 14*14*6.
    conv2 = tf.concat( [tf.contrib.layers.flatten(conv2), l1_fwd], 1)
    
    # Layer 3: Fully Connected. Input = 5*5*16 + 14*14*6. Output = 512.
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

```

A second architecture is based on the Sermanet and LeCun paper. The depth of the first convolutional layer is 108 and the one of the second is 200.


```python
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

```


```python
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


# Shuffle the training data.
from sklearn.utils import shuffle

```


```python
def run(logits, filename):

    EPOCHS = 40
    BATCH_SIZE = 256 

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)

    rate = 0.001
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)

            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

            print("EPOCH {} ...".format(i+1))
            train_accuracy = evaluate(X_train, y_train)
            print("Train Accuracy = {:.3f}".format(train_accuracy))
            validation_accuracy = evaluate(X_valid, y_valid)
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()

        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
        saver.save(sess, filename)
        print("Model saved")
    
```


```python
run(LeNet(x, 0.5), './lenet')
```

Here is an excerpt of the output of the first model training:

.......

EPOCH 37 ...
Train Accuracy = 0.983
Validation Accuracy = 0.959

EPOCH 38 ...
Train Accuracy = 0.986
Validation Accuracy = 0.967

EPOCH 39 ...
Train Accuracy = 0.987
Validation Accuracy = 0.967

EPOCH 40 ...
Train Accuracy = 0.987
Validation Accuracy = 0.969

Test Accuracy = 0.957



```python
run(LeNet2(x, 0.5), './lenet2')
```

Here is an excerpt of the output of the second model training:

.....

EPOCH 37 ...
Train Accuracy = 0.998
Validation Accuracy = 0.975

EPOCH 38 ...
Train Accuracy = 0.997
Validation Accuracy = 0.971

EPOCH 39 ...
Train Accuracy = 0.998
Validation Accuracy = 0.976

EPOCH 40 ...
Train Accuracy = 0.998
Validation Accuracy = 0.966

Test Accuracy = 0.965


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

### Load and Output the Images
I used the images from another Udacity project: https://github.com/darienmt/CarND-TrafficSignClassifier-P2/tree/master/webimages.


```python
### Load the images and plot them here.
### Feel free to use as many code cells as needed.
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline

import glob

names = glob.glob('test-images/*.jpg')
images = [ plt.imread('./' + name ) for name in names ]

fig, axes = plt.subplots(1, len(images), figsize=(10,10),
                         subplot_kw={'xticks': [], 'yticks': []})
   
indexes = range(5)
for ax, index in zip(axes.flat, indexes):
    ax.imshow(images[index])

```


![png](output_36_0.png)



```python
# convert to grayscale
X_data = grayscale(np.array(images))
fig, axes = plt.subplots(1, len(X_data), figsize=(10,10),
                         subplot_kw={'xticks': [], 'yticks': []})
   
indexes = range(5)
for ax, index in zip(axes.flat, indexes):
    ax.imshow(X_data[index], cmap = 'gray')
    
X_data = np.reshape(X_data, (X_data.shape[0],X_data.shape[1],X_data.shape[2], 1))
X_data = (X_data-128)/128
```


![png](output_37_0.png)


### Predict the Sign Type for Each Image


```python
EPOCHS = 40
BATCH_SIZE = 256 

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
logits = LeNet2(x, 1.0)

rate = 0.001
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
# load saved model
import tensorflow as tf
import numpy as np
import csv

label2Name = {}
with open('signnames.csv') as namesFile:
    nameReader = csv.reader(namesFile)
    for row in nameReader:
        label2Name[int(row[0])] = row[1]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./lenet2.meta')
    saver2.restore(sess, './lenet2')
    y = sess.run(logits, feed_dict={x: X_data})
    labels = sess.run(tf.argmax(y,1))
    for i in range(len(labels)):
        print('Label for file', names[i], 'is', label2Name[labels[i]])
```

The accuracy of the first architecture is 80%. The output is:

Label for file ../test-images/road_work.jpg is General caution

Label for file ../test-images/60_kmh.jpg is Speed limit (60km/h)

Label for file ../test-images/left_turn.jpg is Turn left ahead

Label for file ../test-images/yield_sign.jpg is Yield

Label for file ../test-images/stop_sign.jpg is Stop

The accuracy of the second architecture is 80%. The output is:

Label for file ../test-images/road_work.jpg is General caution

Label for file ../test-images/60_kmh.jpg is Speed limit (60km/h)

Label for file ../test-images/left_turn.jpg is Turn left ahead

Label for file ../test-images/yield_sign.jpg is Yield

Label for file ../test-images/stop_sign.jpg is Stop



### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
```

### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./lenet.meta')
    saver2.restore(sess, './lenet')
    y = sess.run(logits, feed_dict={x: X_data})
    top_prob = sess.run(tf.nn.top_k(tf.nn.softmax(y),3))
    for i in range(len(labels)):
        print('For file', names[i], ' top 3 are:')
        for j in range(3):
             print('%2.4f' % (top_prob[0][i][j]), 'for', label2Name[top_prob[1][i][j]])
```

For file ../test-images/road_work.jpg  top 3 are:

0.9994 for General caution

0.0004 for Traffic signals

0.0002 for Pedestrians

For file ../test-images/60_kmh.jpg  top 3 are:

1.0000 for Speed limit (60km/h)

0.0000 for Speed limit (80km/h)

0.0000 for No passing for vehicles over 3.5 metric tons

For file ../test-images/left_turn.jpg  top 3 are:

1.0000 for Turn left ahead

0.0000 for Keep right

0.0000 for No vehicles

For file ../test-images/yield_sign.jpg  top 3 are:

1.0000 for Yield

0.0000 for Speed limit (20km/h)

0.0000 for Speed limit (30km/h)

For file ../test-images/stop_sign.jpg  top 3 are:

1.0000 for Stop

0.0000 for Turn left ahead

0.0000 for Speed limit (60km/h)


```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./lenet2.meta')
    saver2.restore(sess, './lenet2')
    y = sess.run(logits, feed_dict={x: X_data})
    top_prob = sess.run(tf.nn.top_k(tf.nn.softmax(y),3))
    for i in range(len(labels)):
        print('For file', names[i], ' top 3 are:')
        for j in range(3):
             print('%2.4f' % (top_prob[0][i][j]), 'for', label2Name[top_prob[1][i][j]])
```

For file ../test-images/road_work.jpg  top 3 are:

1.0000 for General caution

0.0000 for Traffic signals

0.0000 for Wild animals crossing

For file ../test-images/60_kmh.jpg  top 3 are:

1.0000 for Speed limit (60km/h)

0.0000 for Speed limit (20km/h)

0.0000 for Speed limit (30km/h)

For file ../test-images/left_turn.jpg  top 3 are:

1.0000 for Turn left ahead

0.0000 for Keep right

0.0000 for Stop

For file ../test-images/yield_sign.jpg  top 3 are:

1.0000 for Yield

0.0000 for Speed limit (20km/h)

0.0000 for Speed limit (30km/h)

For file ../test-images/stop_sign.jpg  top 3 are:

1.0000 for Stop

0.0000 for Speed limit (60km/h)

0.0000 for Keep right


### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```
