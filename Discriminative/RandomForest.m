function test_labels = RandomForest(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        Mdl = TreeBagger(10,train_inputs, train_labels_i,'Method','regression');
        test_labels(:,i) = predict(Mdl,test_inputs);
    end