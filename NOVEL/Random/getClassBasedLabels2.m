function test_labels = getClassBasedLabels(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    
    large_metro_idx = (train_inputs(:, 7) == 1);
    medium_metro_idx = (train_inputs(:, 8) == 1);
    small_metro_idx = (train_inputs(:,21)==1);
    non_metro_idx = (train_inputs(:,9)==0);




    %test_labels_11 = glmRandomForest(train_inputs(nchs1_idx, :), train_labels(nchs1_idx, :),test_inputs(test_inputs(:,11) == 1, :));
    %test_labels_12 = glmRidge(train_inputs(nchs1_idx, :), train_labels(nchs1_idx, :),test_inputs(test_inputs(:,11) == 1, :));
    test_labels_13 = glmRandomForest(train_inputs(large_metro_idx, :), train_labels(large_metro_idx, :),test_inputs(test_inputs(:,7) == 1, :));
    test_labels_1 = test_labels_13;
    
    %test_labels_21 = glmRandomForest(train_inputs(nchs2_idx, :), train_labels(nchs2_idx, :),test_inputs(test_inputs(:,11) == 2, :));
    %test_labels_22 = glmRidge(train_inputs(nchs2_idx, :), train_labels(nchs2_idx, :),test_inputs(test_inputs(:,11) == 2, :));
    test_labels_23 = glmRandomForest(train_inputs(medium_metro_idx, :), train_labels(medium_metro_idx, :),test_inputs(test_inputs(:,8) == 1, :));
    test_labels_2 = test_labels_23;
    
    %test_labels_31 = glmRandomForest(train_inputs(nchs3_idx, :), train_labels(nchs3_idx, :),test_inputs(test_inputs(:,11) == 3, :));
    %test_labels_32 = glmRidge(train_inputs(nchs3_idx, :), train_labels(nchs3_idx, :),test_inputs(test_inputs(:,11) == 3, :));
    test_labels_33 = glmRandomForest(train_inputs(small_metro_idx, :), train_labels(small_metro_idx, :),test_inputs(test_inputs(:,21) == 1, :));
    test_labels_3 = test_labels_33;

    %test_labels_41 = glmRandomForest(train_inputs(nchs4_idx, :), train_labels(nchs4_idx, :),test_inputs(test_inputs(:,11) == 4, :));
    %test_labels_42 = glmRidge(train_inputs(nchs4_idx, :), train_labels(nchs4_idx, :),test_inputs(test_inputs(:,11) == 4, :));
    test_labels_43 = glmRandomForest(train_inputs(non_metro_idx, :), train_labels(non_metro_idx, :),test_inputs(test_inputs(:,9) == 0, :));
    test_labels_4 = test_labels_43;
    

    test_labels(test_inputs(:,7) == 1, :) = test_labels_1;
    test_labels(test_inputs(:,8) == 1, :) = test_labels_2;
    test_labels(test_inputs(:,21) == 1, :) = test_labels_3;
    test_labels(test_inputs(:,9) == 0, :) = test_labels_4;
