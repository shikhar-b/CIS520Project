function test_labels = KernelizedNN(train_inputs, train_labels, test_inputs)
    [n,p] = size(train_inputs)
    ids = [1:n];
    ids = ids';
    
    n_fold = 10;
    min_error = intmax;
    
    sigma = [10, 1, 0.1 , 0.01, 0.001, 0.0001];
    indices = crossvalind('Kfold',ids,n_fold);    
    for sig = sigma
        clear Mdl;
        dev_error = 0;
        for i = 1:10
            dev = (indices == i); 
            train = ~dev;

            train_X = train_inputs(train, :); 
            train_Y = train_labels(train, :);
            %ids are predicted

            dev_X =  train_inputs(dev, :);
            dev_result = train_labels(dev, :);
            
            %get prediction labels from predicted ids
            predicted_labels = kernel_regression(train_X,train_Y,dev_X,sigma);

            dev_error = dev_error + error_metric(predicted_labels, dev_result);
        end
        fprintf('K = %i\n', k);
        dev_error = dev_error/n_fold;
        fprintf('dev error = %i\n', dev_error);
        
        if dev_error<min_error
            min_error = dev_error;
            final_sigma = sig;
        end
    end
    
    fprintf('Finalized sigma = %i\n', final_sigma);
    fprintf('dev error = %i\n', min_error);
    
    %TEST
    test_labels = kernel_regression(train_inputs,train_labels,test_inputs,final_sigma);
end

function labels = kernel_regression(Xtrain,Ytrain,Xtest,sigma)
%TODO - Generalize for multiple labels
    % Function that implements kernel regression on the given data (binary classification)
    % Usage: labels = kernel_regression(Xtrain,Ytrain,Xtest)
    
    % Xtrain : N x P Matrix of training data, where N is the number of
    %   training examples, and P is the dimensionality (number of features)
    % Ytrain : N x 1 Vector of training labels (Q dimensions)
    % Xtest : M x P Matrix of testing data, where M is the number of
    %   testing examples.
    % sigma : width of the (gaussian) kernel.
    % labels : return an M x Q vector of predicted labels for testing data.
    % YOUR CODE GOES HERE
    [N, P] = size(Xtrain);
    [M, ~] = size(Xtest);
    labels = zeros(M,1);
    
    % Transform both Xtrain and Xtest into M X P X N matrices for
    % vectorization by repeating and aligning the values
    XtrainRepeated = repmat(reshape(Xtrain', [1 P N]), [M 1 1]);
    XtestRepeated = repmat(Xtest, [1,1,N]);
    
    %Convert labels of Y from 0/1 to -1/1
    YtrainBalanced = (Ytrain * 2) - 1;
    
    % Transform Ytrain into M X 1 X N matrix for
    % vectorization by repeating the values
    YtrainRepeated = repmat(reshape(Ytrain', [1 1 N]), [M 1 1]);
    
    difference = XtrainRepeated - XtestRepeated;
    %Calculate the l2 distance
    %Distance will be a MX1XN dimensional matrix with distance[i][1][j]
    distance = l2(difference);
    
    kernelValue = kernel(distance, sigma);
    
    %Get the labels
    predictedLabelBalanced = sign(sum(kernelValue.*YtrainRepeated, 3));
    labels = floor((predictedLabelBalanced + 1)/2); 
end

function val = kernel(a, sigma)
    val = exp(-(a)/(sigma.^2));
end

function val = l2(a)
    val = sum(a.^2, 2);
end