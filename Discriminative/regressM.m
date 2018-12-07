function test_labels = regressM(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);

    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        b = regress(train_labels_i, train_inputs);
        test_labels(:,i) = test_inputs*b;
        predicted_train_labels(:,i) = train_inputs*b;
    end
