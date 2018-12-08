function test_labels = glmRandomForest(train_inputs, train_labels, test_inputs)
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
    n_comps = 150;
    for i=1:numoutputs
        coeff = pca(train_inputs);
        coeff_selected = coeff(:,1:n_comps);
        train_pre = train_inputs*coeff_selected;
        test_pre = test_inputs*coeff_selected;
        predicted_residuals = RandomForest(train_pre, residual_err(:,i), test_pre);
        test_labels(:,i) = test_labels(:,i) + predicted_residuals;
    end
    