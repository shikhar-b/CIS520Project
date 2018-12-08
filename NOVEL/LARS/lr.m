function test_labels = lr(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        stopCriterion = {};
        stopCriterion{1,1}='maxKernels';
        stopCriterion{1,2} = 100;
        sol = lars(train_labels_i, train_inputs, [], 'lars', stopCriterion);
        [n c] = size(sol);
        x = test_inputs(:,sol(c-10).active_set);
        test_labels(:,i) = x*sol(c-10).beta_OLS'+sol(c-10).b_OLS;
    end
