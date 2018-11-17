function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)

train_inputs = preprocess(train_inputs);
test_inputs  = preprocess(test_inputs);
%pred_labels = GaussianProcessRegression(train_inputs, train_labels, test_inputs);
%pred_labels=randn(size(test_inputs,1),size(train_labels,2));
pred_labels = glm(train_inputs, train_labels, test_inputs);
end

