function pred_labels=predict_labels(train_inputs,train_labels,test_inputs)
    pred_labels_1 = KernelizedNN(train_inputs, train_labels, test_inputs);
    pred_labels_2 = KNN(train_inputs, train_labels, test_inputs);
    pred_labels = 0.5.*pred_labels_1 + 0.5.*pred_labels_2;
end

