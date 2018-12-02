function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

addpath('Analysis');
addpath('Discriminative');
addpath('Generative');
addpath('InstanceBased');
addpath('helper');
addpath('GLM');
addpath('glmnet_matlab');
[train_inputs_pre_r, test_inputs_pre_r] = preprocess(train_inputs, test_inputs, 150);

[train_inputs_pre_g, test_inputs_pre_g] = preprocess(train_inputs, test_inputs, 110);

[train_inputs_pre_ra, test_inputs_pre_ra] = preprocess(train_inputs, test_inputs, 75);


pred_labels_1 = glm(train_inputs, train_labels, test_inputs);
pred_labels_2 = ridgeRegression(train_inputs_pre_r, train_labels, test_inputs_pre_r);
pred_labels_3 = GaussianProcessRegression(train_inputs_pre_g, train_labels, test_inputs_pre_g);
pred_labels_4 = RandomForest(train_inputs_pre_ra, train_labels, test_inputs_pre_ra);
%pred_labels_3 = NNtest(train_inputs, train_labels, test_inputs);
%pred_labels_4 = GRNN(train_inputs, train_labels, test_inputs);
pred_labels = (pred_labels_1+pred_labels_2+pred_labels_3+pred_labels_4)./4;
%pred_labels=randn(size(test_inputs,1),size(train_labels,2));
%pred_labels = glm(train_inputs, train_labels, test_inputs);
end

