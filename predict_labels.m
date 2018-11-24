function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

addpath('Analysis');
addpath('Discriminative');
addpath('Generative');
addpath('InstanceBased');
addpath('helper');
addpath('GLM');
addpath('glmnet_matlab');
[train_inputs_pre, test_inputs_pre] = preprocess(train_inputs, test_inputs);
pred_labels_1 = glm(train_inputs, train_labels, test_inputs);
pred_labels_2 = GaussianProcessRegression(train_inputs_pre, train_labels, test_inputs_pre);
pred_labels_3 = RandomForest(train_inputs_pre, train_labels, test_inputs_pre);
%pred_labels_3 = NNtest(train_inputs, train_labels, test_inputs);
%pred_labels_4 = GRNN(train_inputs, train_labels, test_inputs);
pred_labels = 0.70.*pred_labels_1+0.20.*pred_labels_2+0.10.*pred_labels_3;
%pred_labels=randn(size(test_inputs,1),size(train_labels,2));
%pred_labels = glm(train_inputs, train_labels, test_inputs);
end

