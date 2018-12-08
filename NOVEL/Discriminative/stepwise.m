function test_labels = stepwise(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        [b,SE,PVAL,INMODEL,STATS,NEXTSTEP,HISTORY]  = stepwisefit(train_inputs, train_labels_i);
        b0 = STATS.intercept;
        test_inputs_ones=[ones(size(test_inputs,1),1) test_inputs];
        b = b.*INMODEL';
        beta = [b0;b];
        test_labels(:,i) = test_inputs_ones*beta;
    end
