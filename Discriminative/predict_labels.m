function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)
[train_inputs, test_inputs] = preprocess(train_inputs, test_inputs, 150);
pred_labels = RandomForest(train_inputs, train_labels, test_inputs);
end

