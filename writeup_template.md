# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test-images/60_kmh.jpg "60km Speed Limit"
[image5]: ./test-images/left_turn.jpg "Left Turn"
[image6]: ./test-images/road_work.jpg "Road Work"
[image7]: ./test-images/stop_sign.jpg "Stop Sign"
[image8]: ./test-images/yield_sign.jpg "Yield Sign"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The code is available on [github](https://github.com/ananevans/CarND-Traffic-Sign-Classifier-Project).

I trained and run the exporiments on a computer with three GPU's (NVIDIA Corporation GeForce GTX 1080 Ti). The code with results is available [here](https://github.com/ananevans/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier/Traffic_Sign_Classifier.md)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

* The size of training set is 417,588
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 1)
* The number of unique classes/labels in the data set is 43

I augmented the training data set. The operations and the code are described in Jupiter Notebook.

#### 2. Include an exploratory visualization of the dataset.

A set of 25 grayscale images are available in the saved Jupyter Notebook.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the accuracy is better.

As a last step, I normalized the image data because ...

I decided to generate additional data because the model was overfitting.

To add more data to the the data set, I used the following techniques small rotation, translation, shearing, increase contrast and combinations.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling		   	| 2x2 stride, outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling		   	| 2x2 stride, outputs 5x5x16 					|
| Fully connected		| 5*5*16 + 14*14*6 to 512        				|
| Dropout				| 0.5 rate     									|
| Fully connected		| 512 to 128        							|
| Dropout				| 0.5 rate     									|
| Fully connected		| 128 to 43         							|
| Softmax				|         										|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the AdamOptimizer, a batch size of 256, a 0.001 learning rate, and 40 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.987 
* validation set accuracy of 0.969 
* test set accuracy of 0.957

I started with the LeCun architecture provided in class. First I increased the depth of the convolutional layers. To increase the accuracy, I added the output of the first convolutional layer to the output. The model was overfitting, so I augmented the training set with some combinations of operations. Since the model was still overfitting, I experimented with a different optimizer and various learning rate. This new model was underfitting and had a lower accuracy, so I stayed with the previous attempt.

I also built a model based on the Sermanet and LeCun 2011 paper. It has lower performance than the one reported in the paper, comparable with the one above.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The road work image is misclassified.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Road Work     		| General Caution 								|
| Yield					| Yield											|
| 60 km/h	      		| 60 km/h						 				|
| Left Turn				| Left Turn      								|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The accuracy is much lower than on the test set, due to the small number of images.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code and results are available [here](https://github.com/ananevans/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier/Traffic_Sign_Classifier.md) in the Analyze Performance section.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


