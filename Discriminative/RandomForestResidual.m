function test_labels = RandomForestResidual(train_inputs, train_labels, test_inputs)
    trees = [300 300 300 300 300 300 300 300 300];
    numoutputs = size(train_labels,2);
    [train_inputs_pre_r, test_inputs_pre_r] = preprocess(train_inputs, test_inputs, 150);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        Mdl = TreeBagger(trees(i),train_inputs_pre_r, train_labels_i,'Method','regression');
        test_labels(:,i) = predict(Mdl,test_inputs_pre_r);
        predicted_train_labels(:,i) = predict(Mdl,train_inputs_pre_r);
    end
    residual_err = train_labels - predicted_train_labels;
    [train_inputs_pre, test_inputs_pre] = preprocess(train_inputs, test_inputs, 110);
    predicted_residuals = ridgeRegression(train_inputs_pre, residual_err, test_inputs_pre);
    test_labels = test_labels + predicted_residuals;