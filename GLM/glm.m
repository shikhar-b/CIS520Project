function test_labels = glm(train_inputs, train_labels, test_inputs)
    numoutputs = size(train_labels,2);
    test_labels = zeros(size(test_inputs,1),numoutputs);
    for i=1:numoutputs
        train_labels_i = train_labels(:,i);
        glmMdl = cvglmnet(train_inputs, train_labels_i);
        %cvglmnetPlot(glmMdl);
        %beta = glmMdl.beta;
        %alpha = glmMdl.a0;
        test_labels(:,i) = cvglmnetPredict(glmMdl,test_inputs);
    end
