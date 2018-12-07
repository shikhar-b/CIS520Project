function test_labels = ridgeResidual(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    
    lambda = [50   100   100    50     1   100   100   100   100];
    chosen_components = [75  75   150   125   250    50 60   75 50];
    D = size(train_inputs, 2);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        [train_inputs_pre_r, test_inputs_pre_r] = preprocess(train_inputs, test_inputs, chosen_components(i));
        b = ridge(train_labels_i, train_inputs_pre_r, lambda(i),0);
        test_labels(:,i) = test_inputs_pre_r*b(2:end,:)+b(1,:);
        predicted_train_labels(:,i) = train_inputs_pre_r*b(2:end,:)+b(1,:);
    end
    residual_err = train_labels - predicted_train_labels;
    [train_inputs_pre, test_inputs_pre] = preprocess(train_inputs, test_inputs, 150);
    predicted_residuals = RandomForest(train_inputs_pre, residual_err, test_inputs_pre);
    test_labels = test_labels + predicted_residuals;
