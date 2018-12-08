function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)
addpath('GLM');
addpath('glmnet_matlab');
[train_inputs, test_inputs] = preprocess(train_inputs, test_inputs, 150);
pred_labels = ridgeRegression(train_inputs, train_labels, test_inputs);
end

