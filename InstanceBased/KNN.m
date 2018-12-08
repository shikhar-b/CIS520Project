function test_labels = KNN(train_inputs, train_labels, test_inputs)
    [n,p] = size(train_inputs);
    ids = [1:n];
    ids = ids';
    finalK = 1;
%    n_fold = 10;
%    min_error = intmax;   
%     K = [1 , 3, 5, 7, 9, 11];
%     indices = crossvalind('Kfold',ids,n_fold);    
%     for k = K
%         dev_error = 0;
%         for i = 1:10
%             dev = (indices == i); 
%             train = ~dev;
% 
%             train_X = train_inputs(train, :); 
%             train_Y = ids(train, :);
%             %ids are predicted
% 
%             dev_X =  train_inputs(dev, :);
%             dev_result = train_labels(dev, :);
% 
%             Mdl = fitcknn(train_X,train_Y,'NumNeighbors',k,'Standardize',1);
%             predicted_Y = predict(Mdl,dev_X);
% 
%             %get prediction labels from predicted ids
%             predicted_labels = train_labels(predicted_Y, :);
% 
%             dev_error = dev_error + error_metric(predicted_labels, dev_result);
%         end
%         fprintf('K = %i\n', k);
%         dev_error = dev_error/n_fold;
%         fprintf('dev error = %i\n', dev_error);
%         
%         if dev_error<min_error
%             min_error = dev_error;
%             finalK = k;
%         end
%     end
    
%     fprintf('Finalized K = %i\n', finalK);
%     fprintf('dev error = %i\n', min_error);
%     
    %TEST
    Mdl = fitcknn(train_inputs,ids,'NumNeighbors',finalK,'Standardize',1);
    predicted_ids = predict(Mdl,test_inputs);
    test_labels = train_labels(predicted_ids, :);
    