
data = load('training_data.mat');
train_inputs = data.train_inputs;
train_labels = data.train_labels;
[train_inputs, test_inp_fake] = preprocess(train_inputs, train_inputs,100);
numoutputs = size(train_labels,2);
lambda = [5 10 30 50 100 150 200 250 300 400 500];
lc = 11;
chosen_lambda = zeros(1,numoutputs);
 for i=1:numoutputs
    error_lambda = zeros(1,10);
    for j =1:lc
        lambda_i = lambda(j);
        cv_indices = crossvalind('Kfold',size(train_labels,1),10);
        cv_error = 0;
        for k = 1:10
            xtrain_cv = train_inputs(cv_indices~=k,:);
            ytrain_i_cv = train_labels(cv_indices~=k, i);
            xtest_cv = train_inputs(cv_indices ==k,:);
            ytest_i_cv = train_labels(cv_indices==k, i);
            Mdl = TreeBagger(lambda_i,xtrain_cv, ytrain_i_cv,'Method','regression');
            predicted_y_test = predict(Mdl,xtest_cv);
            cv_error = cv_error + immse(predicted_y_test,ytest_i_cv);
        end
        error_lambda(j) = cv_error/10
    end
    [min_error, ind] = min(error_lambda)
    chosen_lambda(i) = lambda(ind)
 end