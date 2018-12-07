function test_labels = glmRandomForestCV(train_inputs, train_labels, test_inputs)
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
    %[train_inputs_pre_g, test_inputs_pre_g] = determine_components(train_inputs, test_inputs, 150);
    predicted_residuals = zeros(size(test_inputs,1),numoutputs);
    chosen_components=[60 50 10 30 30 30 60 60 50];
    for i=1:numoutputs
        %comp = determine_components(train_inputs, residual_err(:,i), test_inputs);
        [train_inputs_pre_g, test_inputs_pre_g] = preprocess(train_inputs, test_inputs, chosen_components(i));
        predicted_residuals(:,i) = RandomForest(train_inputs_pre_g, residual_err(:,i), test_inputs_pre_g);
        test_labels(:,i) = test_labels(:,i) + predicted_residuals(:,i);
    end
