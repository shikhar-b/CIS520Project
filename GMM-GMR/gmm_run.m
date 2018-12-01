function test_labels = gmm_run(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        
        nbStates = 4;
        Data = [train_inputs,train_labels_i]; 
        %% Training of GMM by EM algorithm, initialized by k-means clustering.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [Priors, Mu, Sigma] = EM_init_kmeans(Data', nbStates);
        [Priors, Mu, Sigma] = EM(Data', Priors, Mu, Sigma);

        %% Use of GMR to retrieve a generalized version of the data and associated
        %% constraints. A sequence of temporal values is used as input, and the 
        %% expected distribution is retrieved. 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [test_label_i(1,:), expSigma] = GMR(Priors, Mu, Sigma,  test_inputs', [1:2021], [1]);
        
        test_labels(:, i) = test_label_i';
    end