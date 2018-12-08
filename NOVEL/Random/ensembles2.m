function test_labels = ensembles2(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    n_comp = [250 150 150 200 170 250 150 150 200];
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        coeff = pca(train_inputs);
        coeff_selected = coeff(:,1:n_comp(i));
        train_pre = train_inputs*coeff_selected;
        test_pre = test_inputs*coeff_selected;
        test_labels_1 = glmRandomForest(train_inputs, train_labels_i, test_inputs);
        train_labels_1 = glmRandomForest(train_inputs, train_labels_i, train_inputs);
        test_labels_2 = glmRidge(train_inputs, train_labels_i, test_inputs);
        train_labels_2 = glmRidge(train_inputs, train_labels_i, train_inputs);
        [B, FitInfo] = lasso(train_pre, train_labels_i,'CV',5,'Alpha',0.5);
         coeff = B(:,FitInfo.IndexMinMSE);
         test_labels_3 = test_pre * coeff + FitInfo.Intercept(FitInfo.IndexMinMSE);
         train_labels_3 = train_pre * coeff + FitInfo.Intercept(FitInfo.IndexMinMSE);
         test_labels_pred = (0.4*test_labels_1+0.4*test_labels_2+0.2*test_labels_3);
         train_labels_pred = (0.40*train_labels_1+0.4*train_labels_2+0.2*train_labels_3);
         residual = train_labels_i-train_labels_pred;
         mdl2 = TreeBagger(400, train_pre, residual,'Method','regression');
         residual_2 = predict(mdl2, test_pre);
         residual_i = residual_2;
         test_labels(:,i) = test_labels_pred+residual_i;
        
    end
        
