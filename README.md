# CIS520Project
Best error achieved on the leaderboard : 0.0732

To run any type of model, go to the corresponding folder and use predict_labels.m to get the predicted labels given the training data, testing data and the training labels. 
main.m in each folder partitions the data into train and test sets and evaluates the model locally. 
To change the selected method in each category, update predict_labels.m to use that method.

### 1. Discriminative Model
Models tried:
Ridge Regression {2nd best discriminative model}
Lasso Regression
Elastic Net (GLM) (Runs on biglab, needs setup on local instance) {Works best individually}
Random Forests
Gaussian Process Regression 
SVM Regression
Shallow Neural Networks

### 2. Generative Model
Models tried:
K means clustering followed by ridge regression in each cluster
GRNN

### 3. Instance Based Model
Models tried:
Kernelized KNN
K nearest neighbors

### 4. Novel
Models tried:
Training error residuals 
Combinations of ensembles
