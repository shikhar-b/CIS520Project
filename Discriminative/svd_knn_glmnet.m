function test_labels = svd_knn_glmnet(train_inputs, train_labels, test_inputs, pc, K, on, sigma)
    X_train = train_inputs;
    Y_train = train_labels;
    X_test = test_inputs;
	
	[Ntrain, ~] = size(X_train);
    
	X_ttl = [X_train;X_test];
    
	dwX_train= X_ttl(1:Ntrain, :);
	dwX_test= X_ttl(Ntrain+1:end, :);
	
	glmnet_obj = glmnet(dwX_train, Y_train, 'gaussian');
	coeff.beta = glmnet_obj.beta(:, 52);
	coeff.alpha = glmnet_obj.a0(52);
	pred_train_price = dwX_train*coeff.beta + coeff.alpha;
	residual = Y_train - pred_train_price;
	
	pred_price = dwX_test*coeff.beta + coeff.alpha;
	
	if on == 1
		[U,D,~] = svdsecon([dwX_train; dwX_test],pc);
		X_trunc = U*D;
		Xtrain_trunc = X_trunc(1:Ntrain, :);
		Xtest_trunc = X_trunc(Ntrain+1:end, :);
		[nnidx,dist] = knnsearch(Xtrain_trunc, Xtest_trunc, 'K', K);
		knn_pred_residual = 0;
        kw = exp(-dist.^2/sigma^2);
        nl = sum(kw, 2);
		for i = 1:size(nnidx, 2)
			knn_pred_residual = knn_pred_residual + kw(:,i) .* residual(nnidx(:,i));
		end
		knn_pred_residual = knn_pred_residual./nl;
        %%%newadd%%
%         m = mean(residual);
%         c = cov(residual);
%         w = 1-normpdf(knn_pred_residual, m, c);
        %%%newadd%%
		Y_pred = pred_price + knn_pred_residual;
	else
		Y_pred = pred_price;
    end
    test_labels = Y_pred;
end