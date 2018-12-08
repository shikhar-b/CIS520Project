# CIS520Project
Best error achieved on the leaderboard : 0.0732

To run any type of model, go to the corresponding folder and use predict_labels.m to get the predicted labels given the training data, testing data and the training labels.   
main.m in each folder partitions the data into train and test sets and evaluates the model locally.  
To change the selected method in each category, update predict_labels.m to use that method.  

Dimensionality reduction done using PCA (Number of components selected using cross validation) (File: preprocess.m)

### 1. Discriminative Model
File: /Discriminative/predict_labels.m
- Methods tried:   
- Ridge Regression {2nd best discriminative model}  (See Files: /Discriminative/ridgeRegression.m)
- Elastic Net (GLM) (Runs on biglab, needs setup on local instance) {Works best individually}  (See Files: /Discriminative/GLM/glm.m)
- Random Forests  (See Files: /Discriminative/RandomForest.m)
- Gaussian Process Regression     (See Files: /Discriminative/GaussianProcessRegression.m)
- SVM Regression  (See Files: /Discriminative/SupportVectorMachines.m)
- Shallow Neural Networks  (See Files: /Discriminative/NNAll.m)

### 2. Instance Based Model
File: /InstanceBased/predict_labels.m
Methods tried: 
- Kernelized KNN   (File: KernelizedNN.m)
- K nearest neighbors   (File: KNN.m)

### 3. Generative Model
File: /Generative/predict_labels.m
Methods tried: 
- K means clustering followed by ridge regression in each cluster (See Files: /Generative/KMeans/ridgeRegression.m, /Generative/KMeans/KM.m)


### 4. Novel 
File: /Novel/predict_labels.m
Methods tried:  
- Training error residuals   
- Combinations of ensembles (see ensembles1.m and ensembles2.m)  (Files: /Random/ensembles1.m and /Random/ensembles2.m) and Gaussian Process Regression  + Residual training (see GPRResidual.m)

