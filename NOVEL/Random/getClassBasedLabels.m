function test_labels = getClassBasedLabels(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    
    non_metro_idx = (train_inputs(:, 9) == 0);
    metro_idx = (train_inputs(:, 9) ~= 0);
    
    test_labels_1 = ridgeResidual(train_inputs(metro_idx, :), train_labels(metro_idx, :),test_inputs(test_inputs(:,9) ~= 0, :));
    test_labels_2 = ridgeResidual(train_inputs(non_metro_idx, :), train_labels(non_metro_idx, :),test_inputs(test_inputs(:,9) == 0, :));
    
    
    test_labels(test_inputs(:,9) ~= 0, :) = test_labels_1;
    test_labels(test_inputs(:,9) == 0, :) = test_labels_2;
    
    

