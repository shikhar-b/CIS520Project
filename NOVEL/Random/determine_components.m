function best_component = determine_components(train_inputs, train_labels, test_inputs)
skip = 500
[n,p] = size(train_inputs)
ids = [1:n];
ids = ids';

n_fold = 10;
min_error = intmax;

%write componenets to test here
K = [10 , 50, 100, 500, 1000, 2000];
indices = crossvalind('Kfold',ids,n_fold);    
for comp = K
    dev_error = 0;
    for i = 1:10
        dev = (indices == i); 
        train = ~dev;

        train_X = train_inputs(train, :); 
        train_Y = train_labels(train, :);
        
        dev_X =  train_inputs(dev, :);
        dev_Y = train_labels(dev, :);

        
        [train_inputs_pre_g, test_inputs_pre_g] = preprocess(train_X, dev_X, comp);
        %write method name here
        predicted_labels = ridgeRegression(train_inputs_pre_g, train_Y, test_inputs_pre_g);
        
        dev_error = dev_error + error_metric(predicted_labels, dev_Y);
     end
    fprintf('components = %i\n', comp);
    dev_error = dev_error/n_fold;
    fprintf('dev error = %i\n', dev_error);

    if dev_error<min_error
        min_error = dev_error;
        final_comp = comp;
    end
  end
    
    fprintf('Finalized comp  = %i\n', final_comp);
    fprintf('dev error = %i\n', min_error);
    best_component = final_comp
%error = zeros(2,uint32(numoutputs/skip));
%plot(error(0,:), comp(1,:));