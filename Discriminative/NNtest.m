function test_labels = NNtest(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    for i=1:numoutputs
        hiddenLayerSize = 10;
        net = fitnet(hiddenLayerSize);
        net.divideParam.trainRatio = 80/100;
        net.divideParam.valRatio = 20/100;
        net.divideParam.testRatio = 0/100;
        train_labels_i = train_labels(:,i);
        [net, tr] = train(net, train_inputs',train_labels_i');
        outputs = net(test_inputs');
        test_labels(:,i) = outputs;
    end