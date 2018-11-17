function test_labels = GaussianProcessRegression(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        gprMdl = fitrgp(train_inputs, train_labels_i);
        test_labels(:,i) = predict(gprMdl,test_inputs);
    end
