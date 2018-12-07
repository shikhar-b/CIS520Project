function test_labels = ridgeRegressionKM(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    train_inputs = [train_inputs ones(size(train_inputs,1),1)];
    test_inputs = [test_inputs ones(size(test_inputs,1),1)];
    lambda = [100  100  100   10    50    50   100   100   100];
    D = size(train_inputs, 2);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        b = ridge(train_labels_i, train_inputs, lambda(i),0);
        test_labels(:,i) = test_inputs*b(2:end,:)+b(1,:);
    end
