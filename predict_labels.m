function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

addpath('Analysis');
addpath('Discriminative');
addpath('Generative');
addpath('InstanceBased');
addpath('helper');
addpath('GLM');
addpath('glmnet_matlab');
[train_inputs, test_inputs] = preprocess(train_inputs, test_inputs);
pred_labels_1 = glm(train_inputs, train_labels, test_inputs);
pred_labels_2 = RandomForest(train_inputs, train_labels, test_inputs);
%pred_labels_3 = NNtest(train_inputs, train_labels, test_inputs);
%pred_labels_4 = GRNN(train_inputs, train_labels, test_inputs);
pred_labels = (0.80.*pred_labels_1 + 0.20.*pred_labels_2);
%pred_labels=randn(size(test_inputs,1),size(train_labels,2));
%pred_labels = glm(train_inputs, train_labels, test_inputs);
end

