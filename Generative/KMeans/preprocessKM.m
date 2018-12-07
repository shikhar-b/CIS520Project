function [train_data, test_data] =preprocessKM(inputs, test_inputs, n_comp)
    %inputs = createPercentageBins(inputs);
    %test_inputs = createPercentageBins(test_inputs);
    inputs = zscore(inputs);
    test_inputs = zscore(test_inputs);
    X_tweets = inputs(:,22:2021);
    X_demo = inputs(:,1:21);
    [coeff, score, latepnt, tsquared, explained] = pca(X_tweets);
    n_comp = min(n_comp, size(coeff,2));
    coeff_selected = coeff(:, 1:n_comp);
    X_tweets_reduced = X_tweets * coeff_selected;
    train_data = [X_demo, X_tweets_reduced];
    test_data = [test_inputs(:,1:21), test_inputs(:,22:2021)*coeff_selected];
