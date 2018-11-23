function test_labels = GRNN(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        ngrnn = newgrnn(train_inputs',train_labels_i',2.0);
        test_labels(:,i) = sim(ngrnn,test_inputs');
    end