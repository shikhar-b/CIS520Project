function test_labels = RandomForest(train_inputs, train_labels, test_inputs)
    trees = [100 300 200 200 200 300 400 400 250];
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        Mdl = TreeBagger(trees(i),train_inputs, train_labels_i,'Method','regression');
        test_labels(:,i) = predict(Mdl,test_inputs);
    end
