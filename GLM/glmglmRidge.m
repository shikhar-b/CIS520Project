function test_labels = glmglmRidge(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    predicted_train_labels = zeros(size(train_inputs,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        glmMdl = cvglmnet(train_inputs, train_labels_i);
        %cvglmnetPlot(glmMdl);
        test_labels(:,i) = cvglmnetPredict(glmMdl,test_inputs);
        predicted_train_labels(:,i) = cvglmnetPredict(glmMdl,train_inputs);
    end
    residual_err = train_labels - predicted_train_labels;
    %[train_inputs_pre_g, test_inputs_pre_g] = preprocess(train_inputs, test_inputs, 110);
    predicted_residuals = glmRidge(train_inputs, residual_err, test_inputs);
    test_labels = test_labels + predicted_residuals;