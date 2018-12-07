function test_labels = Regress(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    train_inputs = [train_inputs ones(size(train_inputs,1),1)];
    test_inputs = [test_inputs ones(size(test_inputs,1),1)];
    
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        b = regress(train_labels_i, train_inputs);
        test_labels(:,i) = test_inputs*b;
        
    end
    
    %residual_err = train_labels - predicted_train_labels;
    %predicted_residuals = zeros(size(test_inputs,1),numoutputs);
    %chosen_components=[60 50 10 30 30 30 60 60 50];
%     for i=1:numoutputs
%         comp = determine_components(train_inputs, residual_err(:,i), test_inputs);
%         [train_inputs_pre_g, test_inputs_pre_g] = preprocess(train_inputs, test_inputs, comp);
%         predicted_residuals(:,i) = RandomForest(train_inputs_pre_g, residual_err(:,i), test_inputs_pre_g);
%         test_labels(:,i) = test_labels(:,i) + predicted_residuals(:,i);
%     end
