# Load pickled data
import pickle
import numpy as np

def load(filename):
    with open(filename, mode='rb') as f:
        data = pickle.load(f)
    inputs, labels = data['features'], data['labels']
    inputs = np.reshape(inputs, (inputs.shape[0],inputs.shape[1],inputs.shape[2], 1))
    return inputs, labels

def load_train():
    return load('/home/ans5k/work/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/augmented-train.p')

def load_valid():
    return load('/home/ans5k/work/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/gray-valid.p')

def load_test():
    return load('/home/ans5k/work/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/gray-test.p')

