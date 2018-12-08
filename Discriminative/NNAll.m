function test_labels = NNAll(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    net=feedforwardnet([10]);
    net.trainParam.epochs = 50;
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 20/100;
    net.divideParam.testRatio = 0/100;
    [net, tr] = train(net, train_inputs',train_labels');
    test_labels =net(test_inputs')';
    