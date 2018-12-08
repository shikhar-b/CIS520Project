function test_labels = Las(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        [B, FitInfo] = lasso(train_inputs, train_labels_i,'CV',10);
        coeff = B(:,FitInfo.IndexMinMSE);
        test_labels(:,i) = test_inputs * coeff + FitInfo.Intercept(FitInfo.IndexMinMSE);
    end
