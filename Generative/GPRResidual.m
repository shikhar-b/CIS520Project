function test_labels = GPRResidual(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    [train_inputs_pre_r, test_inputs_pre_r] = preprocess(train_inputs, test_inputs, 150);
    [train_inputs_pre_g, test_inputs_pre_g] = preprocess(train_inputs, test_inputs, 110);
   
%     lambda = [1e-4 1e-3 5*1e-3 1e-2 5*1e-2 1e-1 0.5 1 5 10];
%     lc = 10;
%     chosen_lambda = zeros(1,numoutputs);
%     for i=1:numoutputs
%         error_lambda = zeros(1,10);
%         for j =1:lc
%             lambda_i = lambda(j);
%             cv_indices = crossvalind('Kfold',size(train_labels,1),10);
%             cv_error = 0;
%             for k = 1:10
%                 xtrain_cv = train_inputs(cv_indices~=k,:);
%                 ytrain_i_cv = train_labels(cv_indices~=k, i);
%                 xtest_cv = train_inputs(cv_indices ==k,:);
%                 ytest_i_cv = train_labels(cv_indices==k, i);
%                 gprMdlCV = fitrgp(xtrain_cv, ytrain_i_cv,'Regularization',lambda_i);
%                 predicted_y_test = predict(gprMdlCV,xtest_cv);
%                 cv_error = cv_error + error_metric(predicted_y_test,ytest_i_cv);
%             end
%             error_lambda(j) = cv_error/10;
%         end
%         [min_error, ind] = min(error_lambda);
%         chosen_lambda(i) = lambda(ind);
%     end
    %chosen_lambda = [ 1e-4    1e-2    1e-5   1e-4    1e-2   1e-2   1e-4    1    1e-2];

    predicted_train_labels = zeros(size(train_inputs_pre_g,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        gprMdl = fitrgp(train_inputs_pre_g, train_labels_i,'Regularization',1e-2);
        test_labels(:,i) = predict(gprMdl,test_inputs_pre_g);
        predicted_train_labels(:,i) = predict(gprMdl,train_inputs_pre_g);
    end
    residual_err = train_labels - predicted_train_labels;
    %[train_inputs_pre_g, test_inputs_pre_g] = preprocess(train_inputs, test_inputs, 110);
    predicted_residuals = RandomForest(train_inputs_pre_g, residual_err, test_inputs_pre_g);
    test_labels = test_labels + predicted_residuals;
