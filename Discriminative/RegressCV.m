function test_labels = RegressCV(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    chosen_components=[40 100 200 250 200 70 100 200 70];
    predicted_train_labels = zeros(size(train_inputs,1),numoutputs);
    for i=1:numoutputs
        %comp = determine_components(train_inputs, train_labels(:,i), test_inputs);
        [train_inputs_pre_g, test_inputs_pre_g] = preprocess(train_inputs, test_inputs, chosen_components(i));
        test_labels(:,i) = Regress(train_inputs_pre_g, train_labels(:,i), test_inputs_pre_g);
        predicted_train_labels(:,i) = Regress(train_inputs_pre_g, train_labels(:,i), train_inputs_pre_g);
    end
    
    residual_err = train_labels - predicted_train_labels;   
    predicted_residuals = zeros(size(test_inputs,1),numoutputs);
    chosen_components=[300 400 400 10 100 300 350 350 400];
     for i=1:numoutputs
         %comp = determine_components(train_inputs, residual_err(:,i), test_inputs);
         [train_inputs_pre_g, test_inputs_pre_g] = preprocess(train_inputs, test_inputs, chosen_components(i));
         predicted_residuals(:,i) = RandomForest(train_inputs_pre_g, residual_err(:,i), test_inputs_pre_g);
         test_labels(:,i) = test_labels(:,i) + predicted_residuals(:,i);
     end