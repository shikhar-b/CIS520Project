
data = load('training_data.mat');
train_inputs = data.train_inputs;
train_labels = data.train_labels;
numoutputs = size(train_labels,2);
n_comp = [10 25 50 75 100 125 150 175 200 250 300 350 400 500 600];

lc = 15;
chosen_comp = zeros(1,numoutputs);
 for i=3:numoutputs
    error_lambda = zeros(1,15);
    for j =1:lc
        [train_inputs_pre, test_inp_fake] = preprocess(train_inputs, train_inputs,n_comp(j));
        lambda_i = 100;
        cv_indices = crossvalind('Kfold',size(train_labels,1),5);
        cv_error = 0;
        for k = 1:5
            xtrain_cv = train_inputs_pre(cv_indices~=k,:);
            ytrain_i_cv = train_labels(cv_indices~=k, i);
            xtest_cv = train_inputs_pre(cv_indices ==k,:);
            ytest_i_cv = train_labels(cv_indices==k, i);
            gprMdl = fitrgp(xtrain_cv, ytrain_i_cv,'Regularization',1e-2);
            predicted_y_test = predict(gprMdl,xtest_cv);
            cv_error = cv_error + immse(predicted_y_test,ytest_i_cv);
        end
        error_lambda(j) = cv_error/10
    end
    [min_error, ind] = min(error_lambda)
    chosen_comp(i) = n_comp(ind)
end