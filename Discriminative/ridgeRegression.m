function test_labels = ridgeRegression(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    lambda = 1e-2;
    D = size(train_inputs, 2);

    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        w = (lambda * eye(D) + train_inputs'*train_inputs)\(train_inputs'*train_labels_i);
        test_labels(:,i) = test_inputs*w;
    end