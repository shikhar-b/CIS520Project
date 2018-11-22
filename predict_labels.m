function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

addpath('Analysis');
addpath('Discriminative');
addpath('Generative');
addpath('InstanceBased');
addpath('helper');
train_inputs = preprocess(train_inputs);
test_inputs  = preprocess(test_inputs);
pred_labels = ridgeRegression(train_inputs, train_labels, test_inputs);
%pred_labels=randn(size(test_inputs,1),size(train_labels,2));
%pred_labels = glm(train_inputs, train_labels, test_inputs);
end

