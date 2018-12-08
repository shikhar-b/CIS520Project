function test_labels = SVMResidual(train_inputs, train_labels, test_inputs)
    lambda = [100.00 0.100 0.0050 0.0500 0.0050 0.0010 0.0500 0.0010 0.0010];
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        gprMdl = fitrsvm(train_inputs, train_labels_i, 'BoxConstraint', lambda(i));
        test_labels(:,i) = predict(gprMdl,test_inputs);
        predicted_train_labels(:,i) = predict(gprMdl, train_inputs);
    end
   residual_err = train_labels - predicted_train_labels;
   predicted_residuals = ridgeRegression(train_inputs, residual_err, test_inputs);
   test_labels = test_labels+predicted_residuals;