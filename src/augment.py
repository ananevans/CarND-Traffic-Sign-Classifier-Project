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